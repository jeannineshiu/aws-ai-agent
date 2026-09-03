"""Graph tests.

All fakes — no API calls, no Chroma, no SQLite, so the suite is free and
deterministic.

Two invariants are load-bearing across phases and are asserted here:
  - rag-only and sql-only stay identical to AWSAgent, so the existing RAG and
    SQL evaluation numbers remain comparable;
  - every loop terminates on its own budget.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.agent.agent import AWSAgent
from src.graph.builder import GraphAgent, build_graph
from src.graph.critic import Verdict
from src.graph.grader import Grade
from src.graph.narrate import describe
from src.graph.supervisor import ContextualPlan, Plan
from src.router.router import RouteType
from src.sql.validate import Review


# ── fakes ─────────────────────────────────────────────────────────────────────

class FakeDoc:
    def __init__(self, text="chunk one"):
        self.page_content = text
        self.metadata = {"title": "Doc", "service": "SageMaker", "source": "u"}


class FakeRouter:
    def __init__(self, route): self.route_value = route
    def route(self, question): return self.route_value


class FakeSupervisor:
    def __init__(self, agents, mode="parallel", refined="REFINED QUESTION",
                 standalone=None):
        self.plan_obj = Plan(agents=agents, mode=mode)
        self.standalone = standalone      # None: the question stands on its own
        self.refined = refined
        self.plan_calls, self.refine_calls = [], []

    def plan(self, question, history=None):
        self.plan_calls.append((question, list(history or [])))
        if self.standalone is None:
            return self.plan_obj
        return ContextualPlan(agents=self.plan_obj.agents, mode=self.plan_obj.mode,
                              standalone_question=self.standalone)

    def refine(self, question, finding, next_agent):
        self.refine_calls.append((question, finding["agent"], next_agent))
        return self.refined


class FakeRAG:
    def __init__(self):
        self.retrieve_calls, self.generate_calls = [], []

    def retrieve(self, query):
        self.retrieve_calls.append(query)
        return [FakeDoc()]

    def generate(self, question, docs):
        self.generate_calls.append(question)
        return {
            "query": question, "answer": "RAG ANSWER",
            "citations": [{"title": "Doc", "service": "SageMaker", "url": "u"}],
            "retrieved_texts": [d.page_content for d in docs],
            "source_count": len(docs),
        }

    def run(self, query):                      # used by v1's AWSAgent
        return self.generate(query, self.retrieve(query))


class FakeSQL:
    """`outcomes` is one entry per execute_sql call: a row list, [] for an empty
    result, or an Exception to raise. The last entry repeats."""

    def __init__(self, outcomes=None):
        self.outcomes = outcomes or [[1]]
        self.generated, self.executed = [], []
        self.conn = None

    def generate_sql(self, question):
        self.generated.append(question)
        return "SELECT 1"

    def validate_sql(self, sql):
        return (True, "OK")

    def execute_sql(self, sql):
        self.executed.append(sql)
        out = self.outcomes[min(len(self.executed) - 1, len(self.outcomes) - 1)]
        if isinstance(out, Exception):
            raise out
        return out

    def explain_results(self, question, df):
        return "SQL ANSWER"

    def run(self, question):                   # used by v1's AWSAgent
        sql = self.generate_sql(question)
        df = self.execute_sql(sql)
        return {"question": question, "answer": self.explain_results(question, df),
                "sql": sql, "data": df, "row_count": len(df)}


class FakeSynthesizer:
    def __init__(self):
        self.merge_calls, self.revise_calls = [], []

    def merge(self, question, rag, sql):
        self.merge_calls.append((question, rag["answer"], sql["answer"]))
        return "MERGED ANSWER"

    def revise(self, question, answer, contexts, critique):
        self.revise_calls.append((answer, critique))
        return "REVISED ANSWER"


class FakeGrader:
    """`verdicts` is one entry per grade() call; the last repeats."""

    def __init__(self, verdicts=(True,), rewritten="BETTER QUERY"):
        self.verdicts = list(verdicts)
        self.rewritten = rewritten
        self.grade_calls, self.rewrite_calls = [], []

    def grade(self, question, docs):
        i = min(len(self.grade_calls), len(self.verdicts) - 1)
        self.grade_calls.append(question)
        ok = self.verdicts[i]
        return Grade(sufficient=ok, missing="" if ok else "no coverage of the feature")

    def rewrite(self, question, query, missing):
        self.rewrite_calls.append((query, missing))
        return self.rewritten


class FakeRepairer:
    def __init__(self):
        self.calls = []

    def repair(self, question, failed_sql, error, conn=None):
        self.calls.append((failed_sql, error))
        return "SELECT 2 -- repaired"


class FakeCritic:
    """`verdicts` is one entry per check() call; the last repeats."""

    def __init__(self, verdicts=(True,)):
        self.verdicts = list(verdicts)
        self.calls = []

    def check(self, question, answer, contexts):
        i = min(len(self.calls), len(self.verdicts) - 1)
        self.calls.append(answer)
        ok = self.verdicts[i]
        return Verdict(grounded=ok, unsupported="" if ok else "invented a number")


def make_graph(agents, mode="parallel", refined="REFINED QUESTION",
               rag=None, sql=None, grader=None, repairer=None, critic=None, **budgets):
    sup = FakeSupervisor(agents, mode, refined)
    rag = rag or FakeRAG()
    sql = sql or FakeSQL()
    syn = FakeSynthesizer()
    agent = GraphAgent(sup, rag, sql, syn, grader, repairer, critic,
                       loops=False, **budgets)
    return agent, sup, rag, sql, syn


# ── topology ──────────────────────────────────────────────────────────────────

def test_graph_nodes():
    g = build_graph(*[object()] * 4).get_graph()
    assert {"supervisor", "rag", "sql", "synthesize", "critic"} <= set(g.nodes)


def test_specialists_report_back_to_the_supervisor():
    edges = {(e.source, e.target) for e in build_graph(*[object()] * 4).get_graph().edges}
    assert ("rag", "supervisor") in edges and ("sql", "supervisor") in edges


def test_the_three_quality_loops_exist():
    """Phase 0 was acyclic by construction. These are the edges that changed that."""
    edges = {(e.source, e.target) for e in build_graph(*[object()] * 4).get_graph().edges}
    assert ("rag", "rag") in edges, "corrective retrieval"
    assert ("sql", "sql") in edges, "SQL repair"
    assert ("critic", "synthesize") in edges, "groundedness redraft"


# ── single-specialist routes must not have moved ──────────────────────────────

@pytest.mark.parametrize("route,agent", [(RouteType.RAG, "rag"), (RouteType.SQL, "sql")])
def test_single_route_still_matches_v1(route, agent):
    v1 = AWSAgent.__new__(AWSAgent)
    v1.router, v1.rag, v1.sql = FakeRouter(route), FakeRAG(), FakeSQL()
    v2, *_ = make_graph([agent])
    q = "What is Amazon Bedrock?"
    assert v1.run(q) == v2.run(q)


def test_single_finding_is_not_sent_to_the_synthesizer():
    v2, _, _, _, syn = make_graph(["rag"])
    v2.run("q")
    assert syn.merge_calls == []


def test_rag_route_does_not_touch_sql():
    v2, _, _, sql, _ = make_graph(["rag"])
    v2.run("q")
    assert sql.generated == []


# ── dispatch ──────────────────────────────────────────────────────────────────

def test_parallel_dispatch_runs_both_on_the_original_question():
    v2, _, rag, sql, _ = make_graph(["rag", "sql"], mode="parallel")
    v2.run("original question")
    assert rag.retrieve_calls == ["original question"]
    assert sql.generated == ["original question"]


def test_parallel_findings_are_merged_not_clobbered():
    v2, _, _, _, syn = make_graph(["rag", "sql"], mode="parallel")
    assert v2.run("q")["route"] == "both"
    assert len(syn.merge_calls) == 1


def test_sequential_feeds_the_first_result_into_the_second_query():
    v2, _, rag, sql, _ = make_graph(["sql", "rag"], mode="sequential",
                                    refined="What does SageMaker do?")
    v2.run("Which service has the most questions and what does it do?")
    assert sql.generated == ["Which service has the most questions and what does it do?"]
    assert rag.retrieve_calls == ["What does SageMaker do?"]


def test_both_route_synthesizes_instead_of_concatenating():
    v2, *_ = make_graph(["rag", "sql"])
    out = v2.run("q")
    assert out["answer"] == "MERGED ANSWER"
    assert "**From documentation:**" not in out["answer"]


def test_route_is_derived_from_who_actually_ran():
    assert make_graph(["rag"])[0].run("q")["route"] == "rag"
    assert make_graph(["sql"])[0].run("q")["route"] == "sql"
    assert make_graph(["rag", "sql"])[0].run("q")["route"] == "both"


def test_dispatch_budget_stops_the_loop():
    sup = FakeSupervisor(["rag", "sql"], "sequential")
    v2 = GraphAgent(sup, FakeRAG(), FakeSQL(), FakeSynthesizer(),
                    loops=False, max_passes=2)
    assert v2.run("q")["answer"]


# ── loop 1: corrective retrieval ──────────────────────────────────────────────

def test_insufficient_retrieval_is_searched_again():
    """The loop this exists for: retrieval that cannot answer the question is
    retried with wording drawn from the documentation, not accepted silently."""
    grader = FakeGrader(verdicts=(False, True), rewritten="SageMaker Model Monitor baseline")
    v2, _, rag, _, _ = make_graph(["rag"], grader=grader)
    v2.run("How does model monitoring work?")

    assert rag.retrieve_calls == ["How does model monitoring work?",
                                  "SageMaker Model Monitor baseline"]
    assert grader.rewrite_calls == [("How does model monitoring work?",
                                     "no coverage of the feature")]


def test_generation_answers_the_original_question_not_the_rewrite():
    """The rewrite is a search term. Answering it instead of the question would
    change the subject."""
    v2, _, rag, _, _ = make_graph(["rag"], grader=FakeGrader(verdicts=(False, True)))
    v2.run("How does model monitoring work?")
    assert rag.generate_calls == ["How does model monitoring work?"]


def test_retrieval_retries_are_capped():
    grader = FakeGrader(verdicts=(False,))          # never satisfied
    v2, _, rag, _, _ = make_graph(["rag"], grader=grader, max_rag_attempts=2)
    out = v2.run("q")
    assert len(rag.retrieve_calls) == 2, "budget not enforced"
    assert out["answer"] == "RAG ANSWER", "must answer from the best it has"


def test_sufficient_retrieval_does_not_retry():
    grader = FakeGrader(verdicts=(True,))
    v2, _, rag, _, _ = make_graph(["rag"], grader=grader)
    v2.run("q")
    assert len(rag.retrieve_calls) == 1 and grader.rewrite_calls == []


def test_no_grader_means_no_grading():
    """loops=False must reproduce the Phase 1 graph exactly."""
    v2, _, rag, _, _ = make_graph(["rag"], grader=None)
    v2.run("q")
    assert len(rag.retrieve_calls) == 1


# ── loop 2: SQL repair ────────────────────────────────────────────────────────

def test_empty_result_triggers_repair():
    """The measured failure: valid SQL, zero rows, and v1 reporting the zero as
    the answer."""
    sql = FakeSQL(outcomes=[[], [169]])
    repairer = FakeRepairer()
    v2, *_ = make_graph(["sql"], sql=sql, repairer=repairer)
    out = v2.run("How many Bedrock questions were asked?")

    assert repairer.calls == [("SELECT 1", "the query ran but found nothing")]
    assert sql.executed == ["SELECT 1", "SELECT 2 -- repaired"]
    assert out["answer"] == "SQL ANSWER" and out["data"] == [169]


def test_execution_error_triggers_repair_with_the_error_text():
    sql = FakeSQL(outcomes=[sqlite_err := Exception("no such column: service"), [7]])
    repairer = FakeRepairer()
    v2, *_ = make_graph(["sql"], sql=sql, repairer=repairer)
    v2.run("q")
    assert repairer.calls == [("SELECT 1", "no such column: service")]


def test_sql_repairs_are_capped():
    sql = FakeSQL(outcomes=[[]])                    # never finds anything
    v2, *_ = make_graph(["sql"], sql=sql, repairer=FakeRepairer(), max_sql_attempts=2)
    out = v2.run("q")
    assert len(sql.executed) == 2, "budget not enforced"
    # The empty result is still reported. Zero is sometimes the true answer, and
    # having tried a repair is not a reason to withhold it.
    assert out["answer"] == "SQL ANSWER"


def test_unrecoverable_error_is_reported_not_hidden():
    """A query that cannot run at all has no result to fall back on."""
    sql = FakeSQL(outcomes=[Exception("no such column: service")])
    v2, *_ = make_graph(["sql"], sql=sql, repairer=FakeRepairer(), max_sql_attempts=2)
    out = v2.run("q")
    assert "Could not answer from the data" in out["answer"]
    assert "no such column: service" in out["answer"]


def test_successful_sql_is_not_repaired():
    sql, repairer = FakeSQL(outcomes=[[1]]), FakeRepairer()
    v2, *_ = make_graph(["sql"], sql=sql, repairer=repairer)
    v2.run("q")
    assert repairer.calls == [] and len(sql.executed) == 1


def test_no_repairer_means_no_repair():
    sql = FakeSQL(outcomes=[[]])
    v2, *_ = make_graph(["sql"], sql=sql, repairer=None)
    v2.run("q")
    assert len(sql.executed) == 1


# ── loop 3: the critic ────────────────────────────────────────────────────────

def test_grounded_answer_is_returned_unchanged():
    critic = FakeCritic(verdicts=(True,))
    v2, _, _, _, syn = make_graph(["rag"], critic=critic)
    out = v2.run("q")
    assert len(critic.calls) == 1 and syn.revise_calls == []
    assert out["answer"] == "RAG ANSWER"


def test_ungrounded_answer_is_redrafted():
    critic = FakeCritic(verdicts=(False, True))
    v2, _, _, _, syn = make_graph(["rag"], critic=critic)
    out = v2.run("q")
    assert syn.revise_calls == [("RAG ANSWER", "invented a number")]
    assert out["answer"] == "REVISED ANSWER"


def test_revisions_are_capped():
    """A critic that never accepts must still terminate."""
    critic = FakeCritic(verdicts=(False,))
    v2, _, _, _, syn = make_graph(["rag"], critic=critic, max_revisions=1)
    out = v2.run("q")
    assert len(syn.revise_calls) == 1, "budget not enforced"
    assert out["answer"] == "REVISED ANSWER"


def test_critic_sees_query_results_not_just_documents():
    """The bug the harder eval set exposed. The critic judged a `both` answer
    against retrieved documents alone, found no support for a figure that came
    from the database, and the redraft replaced a correct answer with "I cannot
    answer" — faithfulness up, answer relevancy down, both for the wrong reason.
    """
    seen = {}

    class RecordingCritic(FakeCritic):
        def check(self, question, answer, contexts):
            seen["contexts"] = contexts
            return super().check(question, answer, contexts)

    v2, *_ = make_graph(["rag", "sql"], critic=RecordingCritic(verdicts=(True,)))
    v2.run("q")

    joined = "\n".join(seen["contexts"])
    assert "chunk one" in joined, "documents missing from the evidence"
    assert "SELECT 1" in joined, "query results missing from the evidence"


def test_critic_still_runs_without_documents():
    """A SQL-only answer is an LLM summarising a DataFrame, which can misread it.
    The query result is evidence, so there is something to check."""
    critic = FakeCritic(verdicts=(True,))
    v2, *_ = make_graph(["sql"], critic=critic)
    v2.run("q")
    assert len(critic.calls) == 1


def test_no_critic_means_no_check():
    v2, _, _, _, syn = make_graph(["rag"], critic=None)
    v2.run("q")
    assert syn.revise_calls == []


# ── concurrency ───────────────────────────────────────────────────────────────

def test_parallel_dispatch_really_runs_concurrently():
    """Fan-out is only worth its complexity if the specialists overlap."""
    import time

    class SlowRAG(FakeRAG):
        def retrieve(self, query):
            time.sleep(0.5)
            return super().retrieve(query)

    class SlowSQL(FakeSQL):
        def execute_sql(self, sql):
            time.sleep(0.5)
            return super().execute_sql(sql)

    v2, *_ = make_graph(["rag", "sql"], mode="parallel", rag=SlowRAG(), sql=SlowSQL())
    t0 = time.perf_counter()
    v2.run("q")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.85, f"specialists did not overlap: {elapsed:.2f}s"


def test_sequential_dispatch_does_not_overlap():
    import time

    class SlowRAG(FakeRAG):
        def retrieve(self, query):
            time.sleep(0.4)
            return super().retrieve(query)

    class SlowSQL(FakeSQL):
        def execute_sql(self, sql):
            time.sleep(0.4)
            return super().execute_sql(sql)

    v2, *_ = make_graph(["sql", "rag"], mode="sequential", rag=SlowRAG(), sql=SlowSQL())
    t0 = time.perf_counter()
    v2.run("q")
    assert time.perf_counter() - t0 >= 0.8


# ── synthesis hygiene ─────────────────────────────────────────────────────────

def test_placeholder_citations_are_stripped():
    """The RAG prompt always asks for a citation, so on an empty retrieval the
    model emits the format string itself. It must not survive as a citation to
    nothing."""
    from src.graph.synthesizer import Synthesizer
    strip = Synthesizer.strip_placeholder_citations
    assert strip("Answer. [Source: <title> | <service>]") == "Answer."
    assert strip("A [Source: <t> | <s>] and B [Source: Real Doc | SageMaker]") == \
        "A and B [Source: Real Doc | SageMaker]"


def test_real_citations_survive():
    from src.graph.synthesizer import Synthesizer
    text = "Answer. [Source: What is AWS Lambda? | Lambda]"
    assert Synthesizer.strip_placeholder_citations(text) == text


# ── SQL repair evidence ───────────────────────────────────────────────────────

def test_probe_terms_extracts_the_literal_that_failed():
    """The literal that matched nothing is what to go looking for."""
    from src.graph.repair import probe_terms
    sql = "SELECT COUNT(*) FROM stackoverflow WHERE tags LIKE '%<bedrock>%' LIMIT 50"
    assert probe_terms(sql) == ["bedrock"]
    assert probe_terms("SELECT 1") == []
    assert probe_terms("WHERE repo LIKE '%aws/sagemaker%' AND state = 'open'") == \
        ["aws/sagemaker", "open"]


def test_repair_evidence_shows_how_the_column_is_really_written():
    """The first version of this sampled arbitrary distinct values, which never
    included the term the query missed — so the model removed the LIMIT instead
    of fixing the spelling. Probing for the failed literal is what makes the
    evidence useful."""
    import sqlite3
    from src.graph.repair import SQLRepairer

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE stackoverflow (tags TEXT)")
    conn.executemany("INSERT INTO stackoverflow VALUES (?)", [
        ("<amazon-web-services><amazon-bedrock>",),
        ("<langchain><amazon-bedrock>",),
        ("<python><amazon-sagemaker>",),
    ])

    block = SQLRepairer.__new__(SQLRepairer)._samples_block(
        conn, "SELECT COUNT(*) FROM stackoverflow WHERE tags LIKE '%<bedrock>%'")
    assert "<amazon-bedrock>" in block
    assert "<python><amazon-sagemaker>" not in block, "unrelated values are noise"


def test_repair_evidence_admits_when_nothing_matches():
    import sqlite3
    from src.graph.repair import SQLRepairer

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE stackoverflow (tags TEXT)")
    conn.execute("INSERT INTO stackoverflow VALUES ('<python>')")

    block = SQLRepairer.__new__(SQLRepairer)._samples_block(
        conn, "SELECT * FROM stackoverflow WHERE tags LIKE '%nonexistent%'")
    assert "Nothing in stackoverflow.tags matches" in block
    assert "<python>" in block


def test_a_count_of_zero_counts_as_finding_nothing():
    """The bug that made the repair loop dead on arrival. `SELECT COUNT(*) ...`
    that matches nothing returns one row containing zero, so a `len(df) == 0`
    check never fires — on exactly the query the loop was built for."""
    import pandas as pd

    class ZeroThenReal(FakeSQL):
        def execute_sql(self, sql):
            self.executed.append(sql)
            return pd.DataFrame({"n": [0]}) if len(self.executed) == 1 \
                else pd.DataFrame({"n": [169]})

    sql, repairer = ZeroThenReal(), FakeRepairer()
    v2, *_ = make_graph(["sql"], sql=sql, repairer=repairer)
    out = v2.run("How many Bedrock questions were asked?")

    assert repairer.calls == [("SELECT 1", "the query ran but found nothing")]
    assert out["data"].iloc[0, 0] == 169


# ── fan-in under desynchronised branches ──────────────────────────────────────

def test_a_retrying_specialist_does_not_lose_the_other_branch():
    """The bug the fakes missed until the real stack hit it.

    Under parallel dispatch each specialist edges back to the supervisor, so it
    is woken once per specialist. Once RAG can loop, the branches stop finishing
    in the same superstep: SQL returns first, wakes the supervisor, and the
    supervisor concludes the turn while RAG is still retrieving. LangGraph does
    not wait — it ends the graph, and RAG's finding is silently dropped.
    """
    grader = FakeGrader(verdicts=(False, True))     # RAG needs two supersteps
    v2, _, rag, sql, syn = make_graph(["rag", "sql"], mode="parallel", grader=grader)
    out = v2.run("q")

    assert len(rag.retrieve_calls) == 2, "RAG should have retried"
    assert len(syn.merge_calls) == 1, "both findings must reach synthesis"
    assert out["route"] == "both"
    assert out["data"] is not None and out["citations"], "neither branch dropped"


def test_supervisor_parks_early_wakeups_without_spending_budget():
    """A parked wake-up must not consume a dispatch pass, or a retry would eat
    the budget meant for dispatching."""
    grader = FakeGrader(verdicts=(False, True))
    v2, *_ = make_graph(["rag", "sql"], mode="parallel", grader=grader, max_passes=2)
    assert v2.run("q")["route"] == "both"


# ── which loops ship ──────────────────────────────────────────────────────────

def test_default_configuration_is_repair_only():
    """Phase 4 measured all three loops. Only SQL repair paid for itself, so it
    is the only one on by default — the grader and the critic stay behind the
    flag rather than in the request path."""
    from unittest.mock import patch
    from src.graph.critic import Critic
    from src.graph.grader import RetrievalGrader
    from src.graph.repair import SQLRepairer

    with patch.object(SQLRepairer, "__init__", return_value=None) as repairer, \
         patch.object(RetrievalGrader, "__init__", return_value=None) as grader, \
         patch.object(Critic, "__init__", return_value=None) as critic:
        GraphAgent(FakeSupervisor(["rag"]), FakeRAG(), FakeSQL(), FakeSynthesizer())
        assert repairer.called, "SQL repair should be on"
        assert not grader.called and not critic.called, "grader and critic should be off"


def test_loops_all_enables_every_loop():
    from unittest.mock import patch
    from src.graph.critic import Critic
    from src.graph.grader import RetrievalGrader
    from src.graph.repair import SQLRepairer

    with patch.object(SQLRepairer, "__init__", return_value=None), \
         patch.object(RetrievalGrader, "__init__", return_value=None) as grader, \
         patch.object(Critic, "__init__", return_value=None) as critic:
        GraphAgent(FakeSupervisor(["rag"]), FakeRAG(), FakeSQL(), FakeSynthesizer(),
                   loops="all")
        assert grader.called and critic.called


# ── multi-turn ────────────────────────────────────────────────────────────────
#
# Everything here needs a real checkpointer. A turn boundary is not a thing the
# graph does; it is a thing that must survive the graph having done nothing to
# clear it, and with no checkpointer there is nothing to survive.

def make_memory_graph(agents, mode="parallel", standalone=None, **kw):
    sup = FakeSupervisor(agents, mode, standalone=standalone)
    rag, sql, syn = FakeRAG(), FakeSQL(), FakeSynthesizer()
    agent = GraphAgent(sup, rag, sql, syn, loops=False, memory=True, **kw)
    return agent, sup, rag, sql, syn


def test_a_finished_turn_is_recorded():
    agent, *_ = make_memory_graph(["rag"])
    agent.run("What is Bedrock?", thread_id="t1")
    history = agent.graph.get_state({"configurable": {"thread_id": "t1"}}).values["history"]
    assert history == [{"question": "What is Bedrock?", "answer": "RAG ANSWER"}]


def test_the_second_turn_sees_the_first():
    agent, sup, *_ = make_memory_graph(["rag"])
    agent.run("What is Bedrock?", thread_id="t1")
    agent.run("How much does it cost?", thread_id="t1")
    _, history = sup.plan_calls[1]
    assert history == [{"question": "What is Bedrock?", "answer": "RAG ANSWER"}]


def test_threads_do_not_see_each_other():
    """Two browser tabs are two conversations, not one."""
    agent, sup, *_ = make_memory_graph(["rag"])
    agent.run("What is Bedrock?", thread_id="t1")
    agent.run("How much does it cost?", thread_id="t2")
    assert sup.plan_calls[1][1] == []


def test_the_first_turn_is_planned_exactly_as_before():
    """No history, no follow-up prompt. The single-turn numbers stay comparable."""
    agent, sup, *_ = make_memory_graph(["rag"])
    agent.run("What is Bedrock?", thread_id="t1")
    assert sup.plan_calls == [("What is Bedrock?", [])]


def test_last_turns_findings_do_not_answer_this_turn():
    """The failure this guards: `findings` accumulates, and a checkpointer keeps
    it. Without the reset, turn two synthesises turn one's evidence - a rag-only
    follow-up would come back routed `both`."""
    agent, sup, rag, sql, syn = make_memory_graph(["rag"])
    agent.run("What is Bedrock?", thread_id="t1")
    sup.plan_obj.agents = ["sql"]
    result = agent.run("How many questions mention it?", thread_id="t1")
    assert result["route"] == "sql"
    assert syn.merge_calls == [], "turn two had one specialist, nothing to merge"


def test_the_dispatch_budget_is_fresh_each_turn():
    """`passes` is a per-turn ceiling. Carried over, turn two starts spent and
    goes straight to synthesis without dispatching anyone."""
    agent, _, rag, _, _ = make_memory_graph(["rag"], max_passes=2)
    for i in range(4):
        agent.run(f"question {i}", thread_id="t1")
    assert len(rag.generate_calls) == 4, "every turn should have reached the specialist"


def test_the_resolved_question_is_what_the_specialist_answers():
    agent, sup, rag, *_ = make_memory_graph(
        ["rag"], standalone="How much does Amazon Bedrock cost?")
    agent.run("What is Bedrock?", thread_id="t1")
    agent.run("How much does it cost?", thread_id="t1")
    assert rag.generate_calls[-1] == "How much does Amazon Bedrock cost?"


def test_the_transcript_keeps_what_the_user_typed():
    """History records the conversation, not the supervisor's reading of it.
    Feeding rewrites back would let one bad resolution compound."""
    agent, *_ = make_memory_graph(
        ["rag"], standalone="How much does Amazon Bedrock cost?")
    agent.run("What is Bedrock?", thread_id="t1")
    agent.run("How much does it cost?", thread_id="t1")
    history = agent.graph.get_state({"configurable": {"thread_id": "t1"}}).values["history"]
    assert [t["question"] for t in history] == ["What is Bedrock?",
                                                "How much does it cost?"]


def test_a_redrafted_answer_is_filed_once():
    """Synthesis runs twice when the critic rejects a draft. The turn is still
    one turn, and only the answer that left is recorded."""
    sup, rag, sql, syn = FakeSupervisor(["rag"]), FakeRAG(), FakeSQL(), FakeSynthesizer()
    agent = GraphAgent(sup, rag, sql, syn, critic=FakeCritic([False, True]),
                       loops=False, memory=True)
    agent.run("What is Bedrock?", thread_id="t1")
    history = agent.graph.get_state({"configurable": {"thread_id": "t1"}}).values["history"]
    assert history == [{"question": "What is Bedrock?", "answer": "REVISED ANSWER"}]


def test_memory_is_off_by_default():
    """A compiled checkpointer makes thread_id mandatory, and the evaluation
    harness runs 30 questions that must not see each other."""
    agent, *_ = make_graph(["rag"])
    assert agent.memory is False
    assert agent.run("q")["route"] == "rag"


# ── narration ─────────────────────────────────────────────────────────────────

def narrate(agent, question, **kw):
    """Every line the app would print for one turn, in order."""
    return [line for node, update in agent.stream_turn(question, **kw)
            if (line := describe(node, update, question))]


def test_a_streamed_turn_says_who_was_dispatched():
    agent, *_ = make_graph(["rag", "sql"], mode="parallel")
    assert "Dispatching documentation + data" in narrate(agent, "q")


def test_narration_shows_the_question_the_user_never_typed():
    """The second half of a sequential plan asks something the supervisor wrote."""
    agent, *_ = make_graph(["sql", "rag"], mode="sequential",
                           refined="What is Amazon SageMaker?")
    lines = narrate(agent, "Which service has the most questions and what does it do?")
    assert any("Asking documentation: *What is Amazon SageMaker?*" in l for l in lines)


def test_narration_shows_what_a_follow_up_was_taken_to_mean():
    """A wrong resolution is the failure mode of multi-turn. The user cannot
    correct one they were never shown."""
    agent, *_ = make_memory_graph(["rag"], standalone="How much does Bedrock cost?")
    agent.run("What is Bedrock?", thread_id="t1")
    lines = narrate(agent, "How much does it cost?", thread_id="t1")
    assert lines[0].startswith("Read as *How much does Bedrock cost?* — Dispatching")


def test_narration_shows_the_repair():
    agent, *_ = make_graph(["sql"], sql=FakeSQL([[], [1]]), repairer=FakeRepairer())
    assert any("repairing it" in l for l in narrate(agent, "q"))


def test_parked_wakeups_and_bookkeeping_are_not_narrated():
    """The supervisor is woken once per specialist and the last node files the
    turn. Neither is a step the user can make sense of."""
    assert describe("supervisor", None, "q") is None
    assert describe("supervisor", {"passes": 3, "awaiting": []}, "q") is None
    assert describe("remember", {"history": [{"question": "q", "answer": "a"}]}, "q") is None


def test_every_narrated_turn_ends_with_the_answer_being_composed():
    for agents, mode in [(["rag"], "parallel"), (["sql"], "parallel"),
                         (["rag", "sql"], "parallel"), (["sql", "rag"], "sequential")]:
        agent, *_ = make_graph(agents, mode=mode)
        lines = narrate(agent, "q")
        assert lines[-1] in ("Composed the answer", "Merged both answers"), (agents, lines)


def test_a_real_check_is_narrated():
    """The other half of the previous test: when the critic does run, say so."""
    sup, rag, sql, syn = FakeSupervisor(["rag"]), FakeRAG(), FakeSQL(), FakeSynthesizer()
    agent = GraphAgent(sup, rag, sql, syn, critic=FakeCritic([True]), loops=False)
    assert "Checked the answer against its sources" in narrate(agent, "q")


def test_streaming_a_turn_leaves_the_same_answer_behind():
    """The app streams the turn for its display, then reads the result from the
    checkpointer instead of reassembling it from the updates it just printed."""
    from src.graph.builder import project

    streamed, *_ = make_memory_graph(["rag", "sql"])
    for _ in streamed.stream_turn("q", thread_id="t1"):
        pass
    invoked, *_ = make_memory_graph(["rag", "sql"])
    assert project("q", streamed.state_of("t1")) == invoked.run("q", thread_id="t2")


# ── the confirmation gate ─────────────────────────────────────────────────────

class ReviewingSQL(FakeSQL):
    """A fake with the three-way reviewer the real pipeline has."""

    def __init__(self, verdicts, outcomes=None):
        super().__init__(outcomes)
        self.verdicts = list(verdicts)          # one per review_sql call, last repeats

    def review_sql(self, sql):
        i = min(len(self.generated) - 1, len(self.verdicts) - 1)
        return Review(self.verdicts[i], f"reason for {self.verdicts[i]}")

    def validate_sql(self, sql):
        v = self.review_sql(sql)
        return v.verdict == "allow", v.reason


def gated(verdicts=("confirm",), outcomes=None, **kw):
    sql = ReviewingSQL(verdicts, outcomes)
    agent = GraphAgent(FakeSupervisor(["sql"]), FakeRAG(), sql, FakeSynthesizer(),
                       loops=False, confirm_sql=True, **kw)
    return agent, sql


def drain(events):
    return [(node, update) for node, update in events]


def test_an_unclassifiable_query_stops_before_it_runs():
    agent, sql = gated(["confirm"])
    drain(agent.stream_turn("q", thread_id="t1"))
    assert sql.executed == [], "the database was touched before anyone was asked"
    assert agent.graph.get_state({"configurable": {"thread_id": "t1"}}).interrupts


def test_approving_runs_the_query_that_was_shown():
    agent, sql = gated(["confirm"])
    drain(agent.stream_turn("q", thread_id="t1"))
    shown = agent.graph.get_state(
        {"configurable": {"thread_id": "t1"}}).interrupts[0].value["sql"]
    drain(agent.resume_turn(True, thread_id="t1"))
    assert sql.executed == [shown]


def test_pausing_does_not_regenerate_the_query():
    """`interrupt()` replays its node from the top. If generation happened
    before the pause, resuming would write a second query and approve the
    first."""
    agent, sql = gated(["confirm"])
    drain(agent.stream_turn("q", thread_id="t1"))
    drain(agent.resume_turn(True, thread_id="t1"))
    assert len(sql.generated) == 1, sql.generated


def test_declining_answers_without_the_query():
    agent, sql = gated(["confirm"])
    drain(agent.stream_turn("q", thread_id="t1"))
    drain(agent.resume_turn(False, thread_id="t1"))
    final = agent.state_of("t1")
    assert sql.executed == []
    assert final["sql_declined"] is True
    assert "not run" in final["answer"]


def test_a_rejected_query_is_not_put_to_a_human():
    """Approving a query the reviewer refuses is not a decision anyone should
    be offered, and the repair loop can still have another go."""
    agent, sql = gated(["reject"], repairer=FakeRepairer())
    drain(agent.stream_turn("q", thread_id="t1"))
    assert not agent.graph.get_state({"configurable": {"thread_id": "t1"}}).interrupts
    assert sql.executed == []


def test_an_ordinary_query_is_not_gated():
    """A blanket confirmation on a read-only database is friction that teaches
    people to click through it."""
    agent, sql = gated(["allow"])
    drain(agent.stream_turn("q", thread_id="t1"))
    assert sql.executed and not agent.graph.get_state(
        {"configurable": {"thread_id": "t1"}}).interrupts


def test_a_repaired_query_is_gated_even_when_it_reads_fine():
    """The repairer rewrites the question's query after it came back empty, and
    the widened filter that turns a zero into a number is exactly what a person
    catches in one look."""
    agent, sql = gated(["allow"], outcomes=[[], [1]], repairer=FakeRepairer())
    drain(agent.stream_turn("q", thread_id="t1"))
    assert len(sql.executed) == 1, "the first attempt runs unasked; the repair does not"
    interrupts = agent.graph.get_state({"configurable": {"thread_id": "t1"}}).interrupts
    assert "rewrite" in interrupts[0].value["reason"]


def test_the_gate_is_off_by_default():
    """The evaluation harness has nobody to ask, and would hang on the first
    query it could not classify."""
    agent, *_ = make_graph(["sql"])
    assert agent.confirm_sql is False
