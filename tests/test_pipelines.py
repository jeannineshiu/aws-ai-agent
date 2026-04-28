import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.sql.pipeline import SQLPipeline
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
