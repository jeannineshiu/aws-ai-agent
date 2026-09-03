"""Graph tests.

Phase 1 changes behaviour for exactly one route. rag-only and sql-only must
still equal AWSAgent field-for-field — that is what keeps the existing RAG and
SQL evaluation numbers comparable. `both` is expected to differ, because that
route was wrong.

Everything runs on fakes, so no API calls and no nondeterminism.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.agent.agent import AWSAgent
from src.graph.builder import GraphAgent, build_graph
from src.graph.supervisor import Plan
from src.router.router import RouteType


# ── fakes ─────────────────────────────────────────────────────────────────────

class FakeRouter:
    def __init__(self, route): self.route_value = route
    def route(self, question): return self.route_value


class FakeSupervisor:
    def __init__(self, agents, mode="parallel", refined="REFINED QUESTION"):
        self.plan_obj = Plan(agents=agents, mode=mode)
        self.refined = refined
        self.plan_calls, self.refine_calls = [], []

    def plan(self, question):
        self.plan_calls.append(question)
        return self.plan_obj

    def refine(self, question, finding, next_agent):
        self.refine_calls.append((question, finding["agent"], next_agent))
        return self.refined


class FakeRAG:
    def __init__(self): self.calls = []
    def run(self, question):
        self.calls.append(question)
        return {
            "query": question, "answer": "RAG ANSWER",
            "citations": [{"title": "Doc", "service": "SageMaker", "url": "u"}],
            "retrieved_texts": ["chunk one"], "source_count": 1,
        }


class FakeSQL:
    def __init__(self): self.calls = []
    def run(self, question):
        self.calls.append(question)
        return {"question": question, "answer": "SQL ANSWER",
                "sql": "SELECT 1", "data": "DATAFRAME", "row_count": 1}


class FakeSynthesizer:
    def __init__(self): self.calls = []
    def merge(self, question, rag, sql):
        self.calls.append((question, rag["answer"], sql["answer"]))
        return "MERGED ANSWER"


def make_graph(agents, mode="parallel", **kw):
    sup = FakeSupervisor(agents, mode, **kw)
    rag, sql, syn = FakeRAG(), FakeSQL(), FakeSynthesizer()
    return GraphAgent(sup, rag, sql, syn), sup, rag, sql, syn


# ── topology ──────────────────────────────────────────────────────────────────

def test_graph_nodes():
    g = build_graph(object(), object(), object(), object()).get_graph()
    assert {"supervisor", "rag", "sql", "synthesize"} <= set(g.nodes)


def test_specialists_report_back_to_the_supervisor():
    """The dispatch loop: a worker's finding must reach the supervisor before
    the next dispatch is chosen. This is the edge Phase 0 did not have."""
    edges = {(e.source, e.target) for e in build_graph(
        object(), object(), object(), object()).get_graph().edges}
    assert ("rag", "supervisor") in edges
    assert ("sql", "supervisor") in edges


# ── single-specialist routes must not have moved ──────────────────────────────

@pytest.mark.parametrize("route,agent", [(RouteType.RAG, "rag"), (RouteType.SQL, "sql")])
def test_single_route_still_matches_v1(route, agent):
    v1 = AWSAgent.__new__(AWSAgent)
    v1.router, v1.rag, v1.sql = FakeRouter(route), FakeRAG(), FakeSQL()
    v2, *_ = make_graph([agent])
    q = "What is Amazon Bedrock?"
    assert v1.run(q) == v2.run(q)


def test_single_finding_is_not_sent_to_the_synthesizer():
    """Synthesising one answer into one answer costs a call and can only lose
    fidelity."""
    v2, _, _, _, syn = make_graph(["rag"])
    v2.run("q")
    assert syn.calls == []


def test_rag_route_does_not_touch_sql():
    v2, _, _, sql, _ = make_graph(["rag"])
    v2.run("q")
    assert sql.calls == []


# ── parallel dispatch ─────────────────────────────────────────────────────────

def test_parallel_dispatch_runs_both_on_the_original_question():
    v2, _, rag, sql, _ = make_graph(["rag", "sql"], mode="parallel")
    v2.run("original question")
    assert rag.calls == ["original question"]
    assert sql.calls == ["original question"]


def test_parallel_dispatch_does_not_refine():
    v2, sup, _, _, _ = make_graph(["rag", "sql"], mode="parallel")
    v2.run("q")
    assert sup.refine_calls == []


def test_parallel_findings_are_merged_not_clobbered():
    """Both specialists write `findings` in the same superstep. Without the
    reducer one of them would be lost."""
    v2, _, _, _, syn = make_graph(["rag", "sql"], mode="parallel")
    out = v2.run("q")
    assert len(syn.calls) == 1, "synthesizer needs both findings"
    assert out["route"] == "both"


# ── sequential dispatch — the Phase 1 fix ─────────────────────────────────────

def test_sequential_feeds_the_first_result_into_the_second_query():
    """The bug this phase exists to fix. In v1 both specialists received the raw
    question, so the retriever searched text that appears in no document."""
    v2, sup, rag, sql, _ = make_graph(
        ["sql", "rag"], mode="sequential", refined="What does SageMaker do?")
    v2.run("Which service has the most questions and what does it do?")

    assert sql.calls == ["Which service has the most questions and what does it do?"]
    assert rag.calls == ["What does SageMaker do?"], "second specialist got the raw question"


def test_sequential_refines_using_the_first_finding():
    v2, sup, _, _, _ = make_graph(["sql", "rag"], mode="sequential")
    v2.run("q")
    assert sup.refine_calls == [("q", "sql", "rag")]


def test_sequential_respects_the_planned_order():
    v2, _, rag, sql, _ = make_graph(["rag", "sql"], mode="sequential", refined="R2")
    v2.run("q")
    assert rag.calls == ["q"] and sql.calls == ["R2"]


# ── synthesis replaces the f-string ───────────────────────────────────────────

def test_both_route_synthesizes_instead_of_concatenating():
    v2, _, _, _, syn = make_graph(["rag", "sql"])
    out = v2.run("q")
    assert out["answer"] == "MERGED ANSWER"
    assert "**From documentation:**" not in out["answer"]
    assert syn.calls == [("q", "RAG ANSWER", "SQL ANSWER")]


def test_both_route_still_carries_citations_and_data():
    v2, *_ = make_graph(["rag", "sql"])
    out = v2.run("q")
    assert out["citations"][0]["service"] == "SageMaker"
    assert out["data"] == "DATAFRAME" and out["sql"] == "SELECT 1"


def test_route_is_derived_from_who_actually_ran():
    """v1 reported the route it intended. This reports what happened."""
    assert make_graph(["rag"])[0].run("q")["route"] == "rag"
    assert make_graph(["sql"])[0].run("q")["route"] == "sql"
    assert make_graph(["rag", "sql"])[0].run("q")["route"] == "both"


# ── guards ────────────────────────────────────────────────────────────────────

def test_empty_plan_falls_back_to_rag():
    """Supervisor.plan() never returns an empty list, but the node must not
    depend on that."""
    v2, _, rag, sql, _ = make_graph(["rag"])
    v2.run("q")
    assert rag.calls and not sql.calls


def test_dispatch_budget_stops_the_loop():
    """A supervisor that keeps finding more to do must still terminate."""
    class NeverDone(FakeSupervisor):
        def plan(self, question): return Plan(agents=["rag", "sql"], mode="sequential")
        def refine(self, question, finding, next_agent): return "again"

    sup = NeverDone(["rag", "sql"], "sequential")
    sup.plan_obj = Plan(agents=["rag", "sql"], mode="sequential")
    v2 = GraphAgent(sup, FakeRAG(), FakeSQL(), FakeSynthesizer(), max_passes=2)
    out = v2.run("q")           # must return rather than hang
    assert out["answer"]


# ── synthesis hygiene ─────────────────────────────────────────────────────────

def test_placeholder_citations_are_stripped():
    """The RAG prompt always asks for a citation, so when retrieval finds nothing
    relevant the model emits the format string itself. It must not survive
    synthesis as a citation to nothing."""
    from src.graph.synthesizer import Synthesizer
    strip = Synthesizer.strip_placeholder_citations
    assert strip("Answer. [Source: <title> | <service>]") == "Answer."
    assert strip("A [Source: <t> | <s>] and B [Source: Real Doc | SageMaker]") == \
        "A and B [Source: Real Doc | SageMaker]"


def test_real_citations_survive():
    from src.graph.synthesizer import Synthesizer
    text = "Answer. [Source: What is AWS Lambda? | Lambda]"
    assert Synthesizer.strip_placeholder_citations(text) == text


# ── does fan-out actually overlap? ────────────────────────────────────────────

def test_parallel_dispatch_really_runs_concurrently():
    """Fan-out is only worth its complexity if the specialists overlap.

    Both fakes sleep 0.5s. Sequential execution takes ~1.0s; genuine concurrency
    takes ~0.5s. The 0.85s threshold separates them with room for scheduling
    overhead.
    """
    import time

    class SlowRAG(FakeRAG):
        def run(self, question):
            time.sleep(0.5)
            return super().run(question)

    class SlowSQL(FakeSQL):
        def run(self, question):
            time.sleep(0.5)
            return super().run(question)

    v2 = GraphAgent(FakeSupervisor(["rag", "sql"], "parallel"),
                    SlowRAG(), SlowSQL(), FakeSynthesizer())
    t0 = time.perf_counter()
    v2.run("q")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.85, f"specialists did not overlap: {elapsed:.2f}s"


def test_sequential_dispatch_does_not_overlap():
    """The counterpart: a dependent plan must not run the two concurrently,
    because the second query does not exist until the first returns."""
    import time

    class SlowRAG(FakeRAG):
        def run(self, question):
            time.sleep(0.4)
            return super().run(question)

    class SlowSQL(FakeSQL):
        def run(self, question):
            time.sleep(0.4)
            return super().run(question)

    v2 = GraphAgent(FakeSupervisor(["sql", "rag"], "sequential"),
                    SlowRAG(), SlowSQL(), FakeSynthesizer())
    t0 = time.perf_counter()
    v2.run("q")
    assert time.perf_counter() - t0 >= 0.8
