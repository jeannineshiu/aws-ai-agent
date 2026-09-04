# src/graph/repair.py
"""Rewrite SQL that failed, using the failure as evidence.

The measured case: asked how many Bedrock questions exist, the model writes
`tags LIKE '%<bedrock>%'`. The real tag is `<amazon-bedrock>`, so the query is
valid SQL that returns zero rows. v1 reports the zero as the answer.

Two things follow. First, an empty result for a question that plainly expects
one is a failure signal, not an answer — so this loop triggers on empty results
as well as on exceptions. Second, the repair needs to see actual values, not
just the schema: no amount of re-reading a column description reveals that tags
are spelled `<amazon-bedrock>`. The repair prompt therefore carries sample
values for the column the failed query touched.
"""
import re
import sqlite3

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.sql.pipeline import DB_SCHEMA

load_dotenv()

REPAIR_PROMPT = ChatPromptTemplate.from_template("""
You are repairing a SQLite query that did not work.

Schema:
{schema}

Question: {question}

The query that failed:
{failed_sql}

What went wrong: {error}

{samples}

Rules:
- Use only SELECT statements
- NEVER select the 'body' column
- Return ONLY the corrected SQL query, no explanation, no markdown, no backticks

If the query found nothing, the usual cause is a literal that does not match how
the data is actually written. Use the values above rather than guessing at the
spelling.

But zero is sometimes the true answer. If the evidence above says nothing in the
column matches, the original query was right and the count really is zero -
return it unchanged. Never widen a filter just to make the number non-zero.

SQL:""")


# Literals inside the failed query. The one that matched nothing is the evidence
# we need to go looking for.
_LITERAL = re.compile(r"'([^']*)'")


def probe_terms(failed_sql: str) -> list[str]:
    """Core search terms taken from the failed query's string literals.

    `'%<bedrock>%'` becomes `bedrock` — the thing to go looking for in the real
    column, so the model can be shown that it is actually written
    `<amazon-bedrock>`.
    """
    terms = []
    for literal in _LITERAL.findall(failed_sql or ""):
        core = literal.strip("%").strip().strip("<>").strip()
        if len(core) > 2 and core.lower() not in terms:
            terms.append(core.lower())
    return terms


def matching_values(conn, table: str, column: str, term: str, limit: int = 6) -> list[str]:
    """Real values in the column that contain the term the query looked for."""
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {column} FROM {table} WHERE {column} LIKE ? LIMIT {limit}",
            (f"%{term}%",),
        ).fetchall()
        return [str(r[0])[:160] for r in rows]
    except sqlite3.Error:
        return []


def sample_values(conn, table: str, column: str, limit: int = 6) -> list[str]:
    """Fallback when nothing matched: show how the column is written at all."""
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {column} FROM {table} "
            f"WHERE {column} IS NOT NULL AND {column} != '' LIMIT {limit}"
        ).fetchall()
        return [str(r[0])[:160] for r in rows]
    except sqlite3.Error:
        return []


def looks_empty(df) -> bool:
    """Did this query find nothing?

    Not the same as "returned no rows". `SELECT COUNT(*) ... WHERE tags LIKE
    '%<bedrock>%'` returns one row containing zero, so a plain len(df) == 0 check
    misses precisely the failure this loop was built for. A single-cell aggregate
    of zero means the filter matched nothing, which is the same event.
    """
    if df is None or len(df) == 0:
        return True
    if getattr(df, "shape", None) == (1, 1):
        value = df.iloc[0, 0]
        if value is None:
            return True
        try:
            return float(value) == 0.0        # float() also handles numpy scalars
        except (TypeError, ValueError):
            return False
    return False


class SQLRepairer:
    # Columns whose literal spelling is the usual cause of an empty result.
    PROBE = (("stackoverflow", "tags"), ("issues", "repo"), ("issues", "labels"))

    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60)

    def _samples_block(self, conn, failed_sql: str) -> str:
        if conn is None:
            return ""
        terms = probe_terms(failed_sql)
        blocks = []
        for table, column in self.PROBE:
            if column not in failed_sql.lower():
                continue
            hits = []
            for term in terms:
                hits.extend(matching_values(conn, table, column, term))
            if hits:
                joined = "\n".join(f"  {v}" for v in dict.fromkeys(hits))
                blocks.append(
                    f"Values in {table}.{column} that actually contain "
                    f"{' or '.join(repr(t) for t in terms)}:\n{joined}"
                )
            else:
                vals = sample_values(conn, table, column)
                if vals:
                    joined = "\n".join(f"  {v}" for v in vals)
                    blocks.append(
                        f"Nothing in {table}.{column} matches "
                        f"{' or '.join(repr(t) for t in terms) or 'the query'}. "
                        f"How the column is actually written:\n{joined}"
                    )
        return "\n\n".join(blocks)

    def repair(self, question: str, failed_sql: str, error: str, conn=None) -> str:
        try:
            response = self.llm.invoke(REPAIR_PROMPT.format_messages(
                schema=DB_SCHEMA,
                question=question,
                failed_sql=failed_sql,
                error=error,
                samples=self._samples_block(conn, failed_sql or ""),
            ))
            sql = (response.content or "").strip()
            if sql.startswith("```"):                 # same unwrapping as generate_sql
                sql = sql.split("```")[1]
                if sql.startswith("sql"):
                    sql = sql[3:]
            sql = sql.strip()
            if sql:
                return sql
        except Exception:
            pass
        return failed_sql
