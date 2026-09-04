"""Multi-turn evaluation harness tests.

The harness is scoring code, and scoring code that is wrong is worse than no
scoring code: it produces a number, and the number is believed. All fakes, no
API calls.

The load-bearing case is the last one: with a supervisor that can only resolve
when it has a history, the `no-history` condition has to score *worse*. If it
does not, the harness is measuring itself rather than the ablation.
"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3

import pandas as pd
import pytest

from scripts.run_multiturn_eval import (
    check_forbidden,
    check_required,
    run_conversation,
    score_answer,
    score_dispatch,
    score_resolution,
    score_value,
    summarise,
)
from src.graph.builder import GraphAgent
from src.graph.supervisor import ContextualPlan

from test_graph import FakeRAG, FakeSQL, FakeSupervisor, FakeSynthesizer


# ── matching ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "How many questions are tagged amazon-rekognition?",
    "How many questions are tagged Amazon Rekognition?",
    "How many questions mention REKOGNITION, though?",
])
def test_a_service_is_named_however_it_is_written(text):
    """Every one of these is a correct rewrite, so none of them may be scored
    as a failure to name the service."""
    assert check_required(text, [["rekognition"]]) == []


def test_a_group_is_satisfied_by_any_of_its_aliases():
    assert check_required("count the questions", [["how many", "count"]]) == []


def test_every_group_has_to_be_satisfied():
    missing = check_required("How many questions are tagged amazon-sagemaker?",
                             [["sagemaker"], ["2024"]])
    assert missing == [["2024"]]


def test_the_previous_subject_leaking_in_is_caught():
    assert check_forbidden("What is AWS Lambda in Amazon Comprehend terms?",
                           ["comprehend"]) == ["comprehend"]


def test_resolution_needs_both_halves():
    item = {"must_mention": [["comprehend"]], "must_not_mention": ["rekognition"]}
    assert score_resolution("How many questions are tagged Comprehend?", item)["ok"]
    # Named the right service and dragged the other one along: still wrong,
    # because the query it produces filters on both.
    both = score_resolution("Comprehend and Rekognition counts?", item)
    assert not both["ok"] and both["leaked"] == ["rekognition"]


def test_a_question_that_resolves_nothing_is_not_scored_as_resolved():
    item = {"must_mention": [["comprehend"]], "must_not_mention": []}
    assert not score_resolution("How many is the second one tagged in?", item)["ok"]


# ── dispatch, answer, value ───────────────────────────────────────────────────

def test_delegation_ignores_order_unless_the_plan_is_sequential():
    parallel = {"expected_agents": ["sql", "rag"], "expected_order": "parallel"}
    assert score_dispatch(["rag", "sql"], parallel)["order_ok"]

    sequential = {"expected_agents": ["sql", "rag"], "expected_order": "sequential"}
    scored = score_dispatch(["rag", "sql"], sequential)
    assert scored["delegation_ok"] and not scored["order_ok"]


def test_an_unlabelled_answer_is_not_scored():
    """`None` drops out of the rate, rather than counting as a failure."""
    assert score_answer("anything", {}) is None


def test_the_answer_check_reads_the_answer_not_the_question():
    item = {"answer_must_mention": [["79"]]}
    assert score_answer("There are 79 such questions.", item)["ok"]
    assert not score_answer("I could not determine the count.", item)["ok"]


def test_value_comparison_runs_the_ground_truth_query():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (n INT)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    item = {"ground_truth_sql": "SELECT COUNT(*) FROM t"}

    assert score_value(pd.DataFrame({"c": [3]}), item, conn)["ok"]
    assert not score_value(pd.DataFrame({"c": [7]}), item, conn)["ok"]
    # A turn that produced no DataFrame at all fails rather than crashing.
    assert not score_value(None, item, conn)["ok"]


def test_a_ranked_name_matches_however_the_label_was_capitalised():
    """The labels are hand-written and their case is arbitrary. A query that
    returns 'SageMaker' where the label says 'sagemaker' ranked the services
    correctly, and exact comparison scored it as a failure - twice, in the
    Phase 5 run."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (name TEXT)")
    conn.execute("INSERT INTO t VALUES ('sagemaker')")
    item = {"ground_truth_sql": "SELECT name FROM t"}

    assert score_value(pd.DataFrame({"c": ["SageMaker"]}), item, conn)["ok"]
    assert score_value(pd.DataFrame({"c": ["  sagemaker "]}), item, conn)["ok"]
    assert not score_value(pd.DataFrame({"c": ["bedrock"]}), item, conn)["ok"]


def test_a_turn_without_ground_truth_sql_is_not_scored_on_value():
    assert score_value(None, {}, None) is None


# ── conditions ────────────────────────────────────────────────────────────────

class FakeCounter:
    """What `run_turn` needs of CallCounter, without patching anything."""
    llm = tokens = 0

    def reset(self):
        pass


class HistoryAwareSupervisor(FakeSupervisor):
    """Resolves the reference only when it has a conversation to resolve against.

    This is the whole subject of the measurement: the words alone are not
    enough, and the ablation has to be able to show that.
    """

    def plan(self, question, history=None):
        self.plan_calls.append((question, list(history or [])))
        resolved = self.standalone if history else question
        return ContextualPlan(
            tasks=[{"agent": a, "query": self.queries.get(a, "")} for a in self.agents],
            mode=self.mode, standalone_question=resolved)


CONVERSATION = {
    "id": "t01",
    "kind": "pronoun",
    "turns": [
        {"question": "What is Amazon Rekognition?", "expected_agents": ["rag"]},
        {"question": "How many questions are tagged with it?",
         "expected_agents": ["rag"],
         "expected_order": "single",
         "expected_standalone": "How many questions are tagged Amazon Rekognition?",
         "must_mention": [["rekognition"]],
         "must_not_mention": ["sagemaker"]},
    ],
}


def make_agent():
    sup = HistoryAwareSupervisor(
        ["rag"], standalone="How many questions are tagged Amazon Rekognition?")
    agent = GraphAgent(sup, FakeRAG(), FakeSQL(), FakeSynthesizer(),
                       loops=False, memory=True)
    return agent, sup


def test_with_history_runs_the_setup_turns_first():
    agent, sup = make_agent()
    row = run_conversation(agent, CONVERSATION, "with-history", FakeCounter(), None)
    assert [q for q, _ in sup.plan_calls] == [t["question"] for t in CONVERSATION["turns"]]
    assert len(row["setup_turns"]) == 1
    assert row["resolution"]["ok"]


def test_no_history_runs_only_the_turn_being_scored():
    agent, sup = make_agent()
    row = run_conversation(agent, CONVERSATION, "no-history", FakeCounter(), None)
    assert [q for q, _ in sup.plan_calls] == [CONVERSATION["turns"][-1]["question"]]
    assert row["setup_turns"] == []


def test_the_ablation_is_what_moves_the_number():
    """Same agent, same question, same labels — only the thread differs."""
    agent, _ = make_agent()
    with_history = run_conversation(agent, CONVERSATION, "with-history",
                                    FakeCounter(), None)
    without = run_conversation(agent, CONVERSATION, "no-history", FakeCounter(), None)
    assert with_history["resolution"]["ok"]
    assert not without["resolution"]["ok"]
    assert without["resolution"]["missing"] == ["rekognition"]


def test_each_condition_gets_its_own_thread():
    """Otherwise the no-history run reads the history the other one just wrote."""
    agent, _ = make_agent()
    run_conversation(agent, CONVERSATION, "with-history", FakeCounter(), None)
    row = run_conversation(agent, CONVERSATION, "no-history", FakeCounter(), None)
    assert not row["resolution"]["ok"]


# ── summary ───────────────────────────────────────────────────────────────────

def row(resolved_ok, control=False, leaked=False):
    return {
        "control": control,
        "resolution": {"ok": resolved_ok, "leaked": ["x"] if leaked else [],
                       "missing": []},
        "dispatch": {"delegation_ok": True, "order_ok": True},
        "answer_check": None, "value_check": None,
        "llm_calls": 4, "tokens": 100, "elapsed": 1.0,
        "setup_llm_calls": 0, "setup_tokens": 0,
    }


def test_controls_are_summarised_apart_from_the_references():
    """A supervisor that resolves everything scores 100% on the follow-ups that
    refer back and 0% on the ones that do not. One combined number hides that."""
    summary = summarise([row(True), row(True), row(False, control=True, leaked=True)])
    assert summary["resolution_accuracy_coref"] == 1.0
    assert summary["resolution_accuracy_control"] == 0.0
    assert summary["over_resolution_rate"] == round(1 / 3, 3)


def test_unscored_checks_do_not_count_as_failures():
    summary = summarise([row(True), row(True)])
    assert summary["answer_accuracy"] is None
    assert summary["value_accuracy"] is None
