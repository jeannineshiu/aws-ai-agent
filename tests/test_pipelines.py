import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3

import pandas as pd
import pytest

from src.sql.validate import review
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.sql.pipeline import SQLPipeline, connect_read_only
from src.rag.pipeline import RAGPipeline
from src.router.router import QueryRouter, RouteType


# ── SQLPipeline.validate_sql ──────────────────────────────────────────────────

def make_sql_pipeline():
    p = SQLPipeline.__new__(SQLPipeline)
    return p


def test_validate_sql_allows_select():
    ok, _ = make_sql_pipeline().validate_sql("SELECT title FROM issues LIMIT 10")
    assert ok is True


def test_validate_sql_blocks_drop():
    ok, msg = make_sql_pipeline().validate_sql("DROP TABLE issues")
    assert ok is False
    assert "DROP" in msg


def test_validate_sql_blocks_insert():
    ok, msg = make_sql_pipeline().validate_sql("INSERT INTO issues VALUES (1)")
    assert ok is False


def test_validate_sql_created_at_is_not_create():
    # 'created_at' must NOT trigger the CREATE keyword block
    ok, _ = make_sql_pipeline().validate_sql(
        "SELECT created_at, title FROM issues LIMIT 5"
    )
    assert ok is True


def test_validate_sql_rejects_non_select():
    ok, msg = make_sql_pipeline().validate_sql("EXEC xp_cmdshell('ls')")
    assert ok is False


# ── the connection: read-only at the engine, not just in review() ─────────────

@pytest.fixture
def db_file(tmp_path):
    path = tmp_path / "issues.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE issues (id INTEGER PRIMARY KEY, title TEXT)")
    conn.execute("INSERT INTO issues (title) VALUES ('endpoint times out')")
    conn.commit()
    conn.close()
    return path


@pytest.mark.parametrize("sql", [
    "SELECT title FROM issues",
    "SELECT COUNT(*) AS n FROM issues WHERE title LIKE '%endpoint%'",
    "SELECT DISTINCT title FROM issues WHERE title IS NOT NULL AND title != '' LIMIT 6",
    "WITH t AS (SELECT title FROM issues) SELECT upper(title) FROM t",
    "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 3) SELECT i FROM n",
])
def test_read_only_connection_still_reads(db_file, sql):
    p = make_sql_pipeline()
    p.conn = connect_read_only(str(db_file))
    assert not p.execute_sql(sql).empty


@pytest.mark.parametrize("sql", [
    "INSERT INTO issues (title) VALUES ('x')",
    "UPDATE issues SET title = 'x'",
    "DELETE FROM issues",
    "DROP TABLE issues",
    "CREATE TABLE t (x)",
    "CREATE TEMP TABLE t (x)",
    "PRAGMA table_info(issues)",
])
def test_a_write_past_review_is_refused_by_sqlite(db_file, sql):
    """Straight to the connection, as a query approved from `confirm` would go."""
    p = make_sql_pipeline()
    p.conn = connect_read_only(str(db_file))
    with pytest.raises(Exception, match="not authorized"):
        p.execute_sql(sql)
    assert sqlite3.connect(db_file).execute("SELECT COUNT(*) FROM issues").fetchone() == (1,)


def test_attach_cannot_create_a_file(db_file, tmp_path):
    """`mode=ro` alone lets this through: it covers the file, not the connection."""
    conn = connect_read_only(str(db_file))
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        conn.execute(f"ATTACH DATABASE '{tmp_path / 'new.db'}' AS other")
    assert not (tmp_path / "new.db").exists()


def test_a_missing_database_is_an_error_not_an_empty_file(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        connect_read_only(str(tmp_path / "missing.db"))
    assert not (tmp_path / "missing.db").exists()


# ── SQLPipeline.explain_results: body truncation ─────────────────────────────

def test_explain_results_drops_body_column():
    p = make_sql_pipeline()
    p.llm = MagicMock()
    p.llm.invoke.return_value = MagicMock(content="summary")

    df = pd.DataFrame({
        "title": ["issue 1"],
        "body": ["x" * 10000],
        "state": ["open"],
    })
    p.explain_results("what are common issues?", df)

    prompt_str = str(p.llm.invoke.call_args)
    assert "x" * 100 not in prompt_str  # body content must not appear in prompt


def test_explain_results_caps_rows():
    p = make_sql_pipeline()
    p.llm = MagicMock()
    p.llm.invoke.return_value = MagicMock(content="summary")

    df = pd.DataFrame({"title": [f"issue {i}" for i in range(100)]})
    p.explain_results("question", df)

    prompt_str = str(p.llm.invoke.call_args)
    assert "issue 20" not in prompt_str  # only first 20 rows should be included


# ── RAGPipeline.format_context ────────────────────────────────────────────────

def make_rag_pipeline():
    p = RAGPipeline.__new__(RAGPipeline)
    return p


def test_format_context_includes_content():
    docs = [
        Document(
            page_content="SageMaker trains models at scale.",
            metadata={"title": "SageMaker Training", "service": "SageMaker", "source": "https://example.com"},
        )
    ]
    context = make_rag_pipeline().format_context(docs)
    assert "SageMaker trains models at scale." in context
    assert "SageMaker Training | SageMaker" in context
    assert "https://example.com" in context


def test_format_context_multiple_docs_separated():
    docs = [
        Document(page_content="doc one", metadata={"title": "A", "service": "S", "source": "u1"}),
        Document(page_content="doc two", metadata={"title": "B", "service": "S", "source": "u2"}),
    ]
    context = make_rag_pipeline().format_context(docs)
    assert "---" in context
    assert "doc one" in context
    assert "doc two" in context


# ── QueryRouter fallback ──────────────────────────────────────────────────────

def make_router():
    r = QueryRouter.__new__(QueryRouter)
    r.llm = MagicMock()
    return r


def test_router_rag():
    r = make_router()
    r.llm.invoke.return_value = MagicMock(content="rag")
    assert r.route("What is Bedrock?") == RouteType.RAG


def test_router_sql():
    r = make_router()
    r.llm.invoke.return_value = MagicMock(content="sql")
    assert r.route("How many issues in 2023?") == RouteType.SQL


def test_router_both():
    r = make_router()
    r.llm.invoke.return_value = MagicMock(content="both")
    assert r.route("Common issues and how to fix them?") == RouteType.BOTH


def test_router_unknown_falls_back_to_rag():
    r = make_router()
    r.llm.invoke.return_value = MagicMock(content="gibberish")
    assert r.route("something weird") == RouteType.RAG


# ── the three-way reviewer ────────────────────────────────────────────────────
#
# Each of these was run against the old single-sweep check before this module
# existed. The first three were rejected, the fourth was allowed.

@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM issues WHERE title LIKE '%delete endpoint%' LIMIT 50",
    "SELECT repo, COUNT(*) FROM issues WHERE body LIKE '%create model%' LIMIT 50",
])
def test_a_keyword_inside_a_string_is_not_a_keyword(sql):
    """People ask this app about deleting endpoints and creating models."""
    assert review(sql).verdict == "allow"


def test_a_common_table_expression_is_a_read():
    assert review("WITH t AS (SELECT repo FROM issues) SELECT * FROM t").verdict == "allow"


def test_two_statements_are_not_one_statement():
    """One was reviewed; the second was going to run behind it."""
    v = review("SELECT * FROM issues LIMIT 50; SELECT * FROM stackoverflow")
    assert v.verdict == "reject" and "2 statements" in v.reason


def test_a_comment_cannot_smuggle_a_keyword_in_or_out():
    assert review("SELECT 1 -- DROP TABLE issues").verdict == "allow"
    assert review("SELECT 1; /* hidden */ DROP TABLE issues").verdict == "reject"


def test_replace_is_a_string_function():
    assert review("SELECT replace(title, 'a', 'b') FROM issues").verdict == "allow"


@pytest.mark.parametrize("sql", ["DROP TABLE issues", "DELETE FROM issues",
                                 "ATTACH DATABASE 'x' AS y", "PRAGMA table_info(issues)"])
def test_writes_and_reaching_outside_the_query_are_refused(sql):
    assert review(sql).verdict == "reject"


def test_what_cannot_be_read_is_asked_about_not_guessed():
    assert review("EXPLAIN SELECT * FROM issues").verdict == "confirm"
    assert review("SELECT * FROM issues WHERE title LIKE '%don't%'").verdict == "confirm"


def test_the_binary_form_refuses_what_it_cannot_put_to_anyone():
    """validate_sql has no human behind it, so the middle tier collapses to no."""
    ok, _ = make_sql_pipeline().validate_sql("EXPLAIN SELECT * FROM issues")
    assert ok is False
