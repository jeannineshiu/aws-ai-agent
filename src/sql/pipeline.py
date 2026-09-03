# src/sql/pipeline.py
import os
import sqlite3
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from src.sql.validate import Review, review

load_dotenv()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "issues.db")

# Describe schema so the LLM knows what tables and columns exist
DB_SCHEMA = """
Database contains two tables:

1. issues — GitHub Issues from AWS ML repositories
   - id: auto increment primary key
   - repo: repository name (e.g. 'aws/sagemaker-python-sdk')
   - issue_number: original GitHub issue number
   - title: issue title
   - body: issue description
   - state: 'open' or 'closed'
   - labels: JSON array of label names
   - comments: number of comments
   - created_at: ISO timestamp
   - closed_at: ISO timestamp or NULL
   - url: GitHub URL

2. stackoverflow — Stack Overflow questions about AWS AI/ML services
   - id: Stack Overflow post ID
   - title: question title
   - body: question body (HTML)
   - tags: pipe-separated tags e.g. '<amazon-sagemaker><python>'
   - score: question score (upvotes - downvotes)
   - view_count: number of views
   - answer_count: number of answers
   - comment_count: number of comments
   - created_at: ISO timestamp
   - closed_at: ISO timestamp or NULL
"""

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert SQL analyst for AWS AI/ML data.
Generate a SQLite SQL query to answer the user's question.

Schema:
{schema}

Rules:
- Use only SELECT statements, never INSERT/UPDATE/DELETE/DROP
- NEVER select the 'body' column — it is too large
- Use LIKE for text search in title, body, tags columns
- For tags in stackoverflow, use: tags LIKE '%<tag-name>%'
- Keep queries simple and efficient; always add LIMIT 50
- Return ONLY the SQL query, no explanation, no markdown, no backticks

Question: {question}

SQL:""")

EXPLANATION_PROMPT = ChatPromptTemplate.from_template("""
You are an AWS AI/ML data analyst.
The user asked: {question}

The SQL query returned this data:
{results}

Write a clear, concise answer in 2-3 sentences.
Highlight the most important insight from the data.
""")


class SQLPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    def generate_sql(self, question: str) -> str:
        """Generate SQL from natural language question."""
        prompt = SQL_GENERATION_PROMPT.format_messages(
            schema=DB_SCHEMA,
            question=question,
        )
        response = self.llm.invoke(prompt)
        sql = response.content.strip()

        # Strip markdown code blocks if LLM added them
        if sql.startswith("```"):
            sql = sql.split("```")[1]
            if sql.startswith("sql"):
                sql = sql[3:]
        return sql.strip()

    def review_sql(self, sql: str) -> Review:
        """Classify the query allow / confirm / reject. See src/sql/validate.py.

        The word-boundary scan this delegates to used to live here and read the
        raw query, so `LIKE '%delete endpoint%'` was a forbidden keyword and
        `WITH ... SELECT` was not a SELECT. Both are questions this app exists
        to answer.
        """
        return review(sql)

    def validate_sql(self, sql: str) -> tuple[bool, str]:
        """The binary form: may this query run with nobody available to ask?

        Only `allow` survives the collapse. A middle tier is a question, and a
        caller that cannot put the question to anyone - v1, or the graph with
        confirmation off - has to answer it the safe way itself. The graph
        lifts this for a query a human has just approved, and only from
        `confirm`; a rejection is never a question.
        """
        verdict = review(sql)
        return verdict.verdict == "allow", ("OK" if verdict.verdict == "allow"
                                            else verdict.reason)

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return results as DataFrame."""
        return pd.read_sql(sql, self.conn)

    def explain_results(self, question: str, df: pd.DataFrame) -> str:
        """Generate natural language explanation of query results."""
        if df.empty:
            results_str = "No results found."
        else:
            # Drop body column if present, cap at 20 rows, truncate long strings
            display_df = df.drop(columns=[c for c in ["body"] if c in df.columns])
            display_df = display_df.head(20)
            results_str = display_df.to_string(index=False, max_colwidth=120)
            results_str = results_str[:4000]  # hard cap to stay well under token limit
        prompt = EXPLANATION_PROMPT.format_messages(
            question=question,
            results=results_str,
        )
        response = self.llm.invoke(prompt)
        return response.content

    def run(self, question: str) -> dict:
        """Run full SQL pipeline: generate → validate → execute → explain."""
        # Generate SQL
        sql = self.generate_sql(question)

        # Validate
        is_valid, reason = self.validate_sql(sql)
        if not is_valid:
            return {
                "question": question,
                "sql": sql,
                "error": reason,
                "answer": f"Could not execute query: {reason}",
                "data": None,
            }

        # Execute
        try:
            df = self.execute_sql(sql)
        except Exception as e:
            return {
                "question": question,
                "sql": sql,
                "error": str(e),
                "answer": f"SQL execution failed: {e}",
                "data": None,
            }

        # Explain
        explanation = self.explain_results(question, df)

        return {
            "question": question,
            "sql": sql,
            "answer": explanation,
            "data": df,
            "row_count": len(df),
        }