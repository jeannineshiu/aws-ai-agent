"""Phase 0 parity tests: the graph must behave exactly like AWSAgent.run().

Everything here runs on fakes — no OpenAI calls, no Chroma, no SQLite — so the
suite stays free and deterministic. The point is to pin the topology and the
output contract, not to test the pipelines (tests/test_pipelines.py does that).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.agent.agent import AWSAgent
from src.graph.builder import GraphAgent, build_graph
from src.graph.nodes import compose_node
from src.router.router import RouteType


# ── fakes ─────────────────────────────────────────────────────────────────────

class FakeRouter:
    def __init__(self, route):
        self.route_value = route
        self.calls = []

    def route(self, question):
        self.calls.append(question)
        return self.route_value


class FakeRAG:
    def __init__(self):
        self.calls = []

    def run(self, question):
        self.calls.append(question)
        return {
            "query": question,
            "answer": "RAG ANSWER",
            "citations": [{"title": "Doc", "service": "SageMaker", "url": "u"}],
            "retrieved_texts": ["chunk one", "chunk two"],
            "source_count": 2,
        }


class FakeSQL:
    def __init__(self):
        self.calls = []

    def run(self, question):
        self.calls.append(question)
        return {
            "question": question,
            "answer": "SQL ANSWER",
            "sql": "SELECT 1",
            "data": "DATAFRAME",
            "row_count": 1,
        }


def make_pair(route):
    """Build a v1 agent and a v2 graph agent sharing identically-behaving fakes."""
    v1 = AWSAgent.__new__(AWSAgent)          # bypass __init__ (it constructs real clients)
    v1.router, v1.rag, v1.sql = FakeRouter(route), FakeRAG(), FakeSQL()

    v2 = GraphAgent(FakeRouter(route), FakeRAG(), FakeSQL())
    return v1, v2


# ── topology ──────────────────────────────────────────────────────────────────

def test_graph_has_expected_nodes():
    g = build_graph(object(), object(), object()).get_graph()
    assert {"route", "rag", "sql", "compose"} <= set(g.nodes)


def test_every_edge_points_forward():
    """Phase 0 is acyclic by construction. Cycles are Phase 2."""
    g = build_graph(object(), object(), object()).get_graph()
    order = ["__start__", "route", "rag", "sql", "compose", "__end__"]
    for e in g.edges:
        assert order.index(e.source) < order.index(e.target), f"back-edge: {e.source}->{e.target}"


# ── per-route parity with v1 ──────────────────────────────────────────────────

@pytest.mark.parametrize("route", [RouteType.RAG, RouteType.SQL, RouteType.BOTH])
def test_graph_output_matches_v1(route):
    v1, v2 = make_pair(route)
    q = "What is Amazon Bedrock?"
    assert v1.run(q) == v2.run(q)


def test_rag_route_does_not_touch_sql():
    v2 = GraphAgent(FakeRouter(RouteType.RAG), FakeRAG(), sql := FakeSQL())
    v2.run("q")
    assert sql.calls == []


def test_sql_route_does_not_touch_rag():
    v2 = GraphAgent(FakeRouter(RouteType.SQL), rag := FakeRAG(), FakeSQL())
    v2.run("q")
    assert rag.calls == []


def test_both_route_runs_rag_before_sql():
    rag, sql = FakeRAG(), FakeSQL()
    GraphAgent(FakeRouter(RouteType.BOTH), rag, sql).run("q")
    assert rag.calls == ["q"] and sql.calls == ["q"]


def test_both_route_keeps_the_v1_concatenation():
    out = GraphAgent(FakeRouter(RouteType.BOTH), FakeRAG(), FakeSQL()).run("q")
    assert out["answer"] == (
        "**From documentation:**\nRAG ANSWER\n\n"
        "**From data analysis:**\nSQL ANSWER"
    )


def test_both_route_carries_citations_and_data():
    out = GraphAgent(FakeRouter(RouteType.BOTH), FakeRAG(), FakeSQL()).run("q")
    assert out["citations"][0]["service"] == "SageMaker"
    assert out["data"] == "DATAFRAME"
    assert out["sql"] == "SELECT 1"


def test_unknown_route_falls_back_to_rag():
    """QueryRouter maps unparseable output to RAG; the graph must not diverge."""
    v2 = GraphAgent(FakeRouter("nonsense"), rag := FakeRAG(), sql := FakeSQL())
    out = v2.run("q")
    assert rag.calls == ["q"] and sql.calls == []
    assert out["answer"] == "RAG ANSWER"


# ── state plumbing ────────────────────────────────────────────────────────────

def test_findings_reducer_accumulates_rather_than_replaces():
    """The guard that makes Phase 1's parallel fan-out safe."""
    from src.graph.state import AgentState
    from typing import get_type_hints, Annotated, get_args
    import operator

    hint = get_type_hints(AgentState, include_extras=True)["findings"]
    assert operator.add in get_args(hint), "findings must be an accumulating channel"


def test_compose_handles_empty_findings():
    out = compose_node({"question": "q", "findings": []})
    assert "Unable to process" in out["answer"]
    assert out["citations"] == [] and out["data"] is None and out["sql"] is None
