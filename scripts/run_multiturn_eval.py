# scripts/run_multiturn_eval.py
"""Phase 4, for the second turn.

Phase 4 measured single questions and cut two of three quality loops on the
numbers. Phase 3 then added multi-turn, and nothing measured it: the supervisor
resolves "how much does it cost" into a standalone question, and whether it
resolves it *correctly* was argued from a handful of examples typed into the
Streamlit box.

Wrong resolution is the failure mode of multi-turn, and it is a quiet one. The
graph still dispatches, still retrieves, still answers - it answers a different
question. Route-level metrics cannot see it: "how many questions are tagged with
it" goes to `sql` whether `it` became Rekognition or SageMaker.

So this scores the second turn on four things, in the order the failure
propagates:

    resolution   did the standalone question name the thing referred to,
                 and nothing it was not referred to
    delegation   were the right specialists dispatched, in the right relation
    answer       does the answer contain what the resolved question asked for
    value        for turns with ground-truth SQL, does the number match

and runs the whole set twice:

    with-history   turn one, then turn two on the same thread
    no-history     turn two alone on a fresh thread

The second condition is the ablation, and it is also what every version before
Phase 3 did with a follow-up. The difference between the columns is what the
checkpointer and the follow-up prompt bought.

    python scripts/run_multiturn_eval.py                  # both conditions
    python scripts/run_multiturn_eval.py --condition with-history
    python scripts/run_multiturn_eval.py --judge          # + LLM equivalence

Resolution is scored by required and forbidden mentions rather than by string
equality with `expected_standalone`: there are many correct rewrites of "and
Rekognition?" and exactly one of them is the one a human wrote down. The
labelled rewrite is kept for reading, and for `--judge`, which asks a model
whether the two are equivalent - a second opinion with a known noise floor,
not the headline number.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re
import sqlite3
from datetime import datetime

import pandas as pd

from scripts.run_evaluation import CallCounter, first_value, values_match

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(_ROOT, "data", "processed", "issues.db")
DATASET = os.path.join(_ROOT, "data", "processed", "ground_truth_multiturn.json")

CONDITIONS = ("with-history", "no-history")


# ── matching ──────────────────────────────────────────────────────────────────
#
# Everything below is deterministic on purpose. The judge is optional and its
# verdict is reported separately; a metric that needs an LLM to reproduce is a
# metric with the ±0.05 noise band the RAGAS numbers already carry.

def _norm(text: str) -> str:
    """Lowercase, and punctuation that separates words treated as space.

    `amazon-rekognition`, `Amazon Rekognition` and `rekognition,` all have to
    count as naming the same service, because all three are rewrites a correct
    supervisor might produce.
    """
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower())


def mentions(text: str, alias: str) -> bool:
    return _norm(alias) in _norm(text)


def check_required(text: str, groups: list[list[str]]) -> list[list[str]]:
    """Groups none of whose aliases appear. Empty means every group is satisfied.

    A group is a set of ways to say the same thing; the rewrite has to say each
    thing some way.
    """
    return [g for g in (groups or []) if not any(mentions(text, a) for a in g)]


def check_forbidden(text: str, terms: list[str]) -> list[str]:
    """Terms that leaked in. For a follow-up about a new subject, the previous
    subject appearing at all is the over-resolution failure."""
    return [t for t in (terms or []) if mentions(text, t)]


def score_resolution(resolved: str, item: dict) -> dict:
    missing = check_required(resolved, item.get("must_mention"))
    leaked = check_forbidden(resolved, item.get("must_not_mention"))
    return {
        "resolved": resolved,
        "missing": ["|".join(g) for g in missing],
        "leaked": leaked,
        "ok": not missing and not leaked,
    }


def score_dispatch(agents: list[str], item: dict) -> dict:
    expected = item.get("expected_agents") or []
    exact = set(agents) == set(expected)
    ordered = exact and (item.get("expected_order") != "sequential"
                         or agents == expected)
    return {"agents": agents, "expected_agents": expected,
            "delegation_ok": exact, "order_ok": ordered}


def score_answer(answer: str, item: dict) -> dict | None:
    """Whether the answer carries what the resolved question asked for.

    Resolution is the mechanism; this is the outcome. They come apart: a turn
    can resolve correctly and still lose the number in synthesis, and a turn
    can resolve wrongly and still mention the right service in passing.
    """
    groups = item.get("answer_must_mention")
    if not groups:
        return None
    missing = check_required(answer, groups)
    return {"missing": ["|".join(g) for g in missing], "ok": not missing}


def score_value(data, item: dict, conn) -> dict | None:
    """The generated query's first cell against the ground truth's, 5% tolerance
    on numbers - the same comparison the single-turn harness makes."""
    if not item.get("ground_truth_sql"):
        return None
    try:
        expected = first_value(pd.read_sql(item["ground_truth_sql"], conn))
    except Exception as e:
        return {"ok": False, "expected": None, "got": None, "error": str(e)}
    got = first_value(data) if isinstance(data, pd.DataFrame) else None
    return {"ok": values_match(got, expected), "expected": expected, "got": got}


# ── running ───────────────────────────────────────────────────────────────────

def run_turn(agent, question: str, thread_id: str, counter) -> dict:
    counter.reset()
    state = agent.run_traced(question, thread_id=thread_id)
    findings = state.get("findings", [])
    return {
        "question": question,
        "resolved_question": state.get("resolved_question") or question,
        "answer": state.get("answer", ""),
        "route": state.get("route", ""),
        "agents": [f["agent"] for f in findings],
        "data": state.get("data"),
        "sql": state.get("sql"),
        "trajectory": state.get("trajectory", []),
        "elapsed": state.get("elapsed", 0.0),
        "llm_calls": counter.llm,
        "tokens": counter.tokens,
    }


def run_conversation(agent, conv: dict, condition: str, counter, conn) -> dict:
    """One conversation under one condition. Only the last turn is scored.

    `no-history` skips every turn but the last and gives it a thread of its own,
    which is exactly the state a pre-Phase-3 agent was in when a user typed a
    follow-up: the words, and nothing else.
    """
    thread = f"{conv['id']}-{condition}"
    turns = conv["turns"]
    setup = []
    if condition == "with-history":
        for turn in turns[:-1]:
            setup.append(run_turn(agent, turn["question"], thread, counter))

    item = turns[-1]
    result = run_turn(agent, item["question"], thread, counter)

    return {
        "id": conv["id"],
        "kind": conv.get("kind", ""),
        "control": bool(conv.get("control")),
        "depends_on_turn1": bool(conv.get("depends_on_turn1")),
        "condition": condition,
        "setup_turns": [{k: t[k] for k in ("question", "answer", "route")}
                        for t in setup],
        "expected_standalone": item.get("expected_standalone", ""),
        "resolution": score_resolution(result["resolved_question"], item),
        "dispatch": score_dispatch(result["agents"], item),
        "answer_check": score_answer(result["answer"], item),
        "value_check": score_value(result["data"], item, conn),
        "answer": result["answer"],
        "sql": result["sql"],
        "trajectory": result["trajectory"],
        # The cost of the scored turn, plus what the setup turns cost to get
        # there. Both matter: the follow-up prompt is bigger, and the condition
        # that has history had to run a turn to have it.
        "llm_calls": result["llm_calls"],
        "tokens": result["tokens"],
        "elapsed": round(result["elapsed"], 2),
        "setup_llm_calls": sum(t["llm_calls"] for t in setup),
        "setup_tokens": sum(t["tokens"] for t in setup),
        "setup_elapsed": round(sum(t["elapsed"] for t in setup), 2),
    }


# ── optional second opinion ───────────────────────────────────────────────────

def judge_equivalence(rows: list[dict]) -> None:
    """Ask a model whether the rewrite means what the labelled rewrite means.

    Writes `equivalent` onto each row. Kept out of the headline metrics: it
    disagrees with the deterministic check exactly where the labelled rewrite
    was one of several correct ones, which is informative but not a score.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class Verdict(BaseModel):
        equivalent: bool = Field(description="Do the two questions ask for the same thing?")
        why: str = Field(description="One sentence.")

    prompt = ChatPromptTemplate.from_template(
        "Two rewrites of the same follow-up question in a conversation.\n\n"
        "Reference: {expected}\nCandidate: {actual}\n\n"
        "Do they ask for the same thing? Wording, level of formality and the "
        "exact form of a tag do not matter. The subject, the metric and any "
        "constraint such as a year do.")
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                       timeout=30).with_structured_output(Verdict)

    for row in rows:
        if not row["expected_standalone"]:
            continue
        try:
            v = judge.invoke(prompt.format_messages(
                expected=row["expected_standalone"],
                actual=row["resolution"]["resolved"]))
            row["equivalent"] = bool(v.equivalent)
            row["equivalent_why"] = v.why
        except Exception as e:
            row["equivalent"] = None
            row["equivalent_why"] = str(e)


# ── report ────────────────────────────────────────────────────────────────────

def rate(rows: list[dict], key) -> float | None:
    values = [key(r) for r in rows]
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarise(rows: list[dict]) -> dict:
    coref = [r for r in rows if not r["control"]]
    control = [r for r in rows if r["control"]]
    return {
        "samples": len(rows),
        "resolution_accuracy": rate(rows, lambda r: r["resolution"]["ok"]),
        # Split because the two halves fail in opposite directions: a follow-up
        # that refers back fails by not resolving, a follow-up that does not
        # fails by resolving anyway.
        "resolution_accuracy_coref": rate(coref, lambda r: r["resolution"]["ok"]),
        "resolution_accuracy_control": rate(control, lambda r: r["resolution"]["ok"]),
        "over_resolution_rate": rate(rows, lambda r: bool(r["resolution"]["leaked"])),
        "delegation_accuracy": rate(rows, lambda r: r["dispatch"]["delegation_ok"]),
        "order_accuracy": rate(rows, lambda r: r["dispatch"]["order_ok"]),
        "answer_accuracy": rate(rows, lambda r: r["answer_check"]["ok"]
                                if r["answer_check"] else None),
        "value_accuracy": rate(rows, lambda r: r["value_check"]["ok"]
                               if r["value_check"] else None),
        "judged_equivalent": rate(rows, lambda r: r.get("equivalent")),
        "llm_calls_per_turn": round(sum(r["llm_calls"] for r in rows) / max(len(rows), 1), 2),
        "tokens_per_turn": round(sum(r["tokens"] for r in rows) / max(len(rows), 1)),
        "seconds_per_turn": round(sum(r["elapsed"] for r in rows) / max(len(rows), 1), 2),
        "setup_llm_calls_total": sum(r["setup_llm_calls"] for r in rows),
        "setup_tokens_total": sum(r["setup_tokens"] for r in rows),
    }


METRIC_LABELS = [
    ("resolution_accuracy", "resolution", "pct"),
    ("resolution_accuracy_coref", "  refers back", "pct"),
    ("resolution_accuracy_control", "  new subject", "pct"),
    ("over_resolution_rate", "over-resolution", "pct"),
    ("delegation_accuracy", "delegation", "pct"),
    ("order_accuracy", "order", "pct"),
    ("answer_accuracy", "answer contains", "pct"),
    ("value_accuracy", "SQL value", "pct"),
    ("judged_equivalent", "judge: equivalent", "pct"),
    ("llm_calls_per_turn", "LLM calls / turn", "num"),
    ("tokens_per_turn", "tokens / turn", "int"),
    ("seconds_per_turn", "seconds / turn", "num"),
]


def fmt(value, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "int":
        return f"{value:,}"
    return f"{value:.2f}"


def print_report(report: dict):
    summaries = report["summaries"]
    order = [c for c in CONDITIONS if c in summaries]

    print("\n" + "=" * 72)
    print("MULTI-TURN EVALUATION — second turn of each conversation")
    print("=" * 72)
    print(f"\n{'':<22}" + "".join(f"{c:>16}" for c in order) +
          ("       delta" if len(order) == 2 else ""))

    for key, label, kind in METRIC_LABELS:
        values = [summaries[c].get(key) for c in order]
        if all(v is None for v in values):
            continue
        line = f"{label:<22}" + "".join(f"{fmt(v, kind):>16}" for v in values)
        if len(order) == 2 and None not in values:
            delta = summaries["with-history"][key] - summaries["no-history"][key]
            sign = "+" if delta >= 0 else ""
            line += f"   {sign}{fmt(delta, kind) if kind != 'pct' else f'{delta:.0%}'}"
        print(line)

    if len(order) == 2:
        setup = summaries["with-history"]
        print(f"\nturn one cost, paid to have a history at all: "
              f"{setup['setup_llm_calls_total']} calls, "
              f"{setup['setup_tokens_total']:,} tokens")

    print("\n--- per conversation ---")
    for condition in order:
        print(f"\n  {condition}")
        for row in report["rows"]:
            if row["condition"] != condition:
                continue
            marks = "".join([
                "R" if row["resolution"]["ok"] else ".",
                "D" if row["dispatch"]["delegation_ok"] else ".",
                "O" if row["dispatch"]["order_ok"] else ".",
                "A" if (row["answer_check"] or {}).get("ok") else
                ("-" if row["answer_check"] is None else "."),
                "V" if (row["value_check"] or {}).get("ok") else
                ("-" if row["value_check"] is None else "."),
            ])
            tag = "control" if row["control"] else row["kind"]
            print(f"    {row['id']}  {marks}  {tag:<16} "
                  f"{row['resolution']['resolved'][:60]!r}")
            problems = []
            if row["resolution"]["missing"]:
                problems.append(f"never named {row['resolution']['missing']}")
            if row["resolution"]["leaked"]:
                problems.append(f"dragged in {row['resolution']['leaked']}")
            if not row["dispatch"]["delegation_ok"]:
                problems.append(f"dispatched {row['dispatch']['agents']} "
                                f"want {row['dispatch']['expected_agents']}")
            if row["value_check"] and not row["value_check"]["ok"]:
                problems.append(f"value {row['value_check']['got']} "
                                f"want {row['value_check']['expected']}")
            if problems:
                print(f"          {'; '.join(problems)}")
    print("\n  R resolution  D delegation  O order  A answer  V value  "
          "(- not applicable)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="both",
                        choices=[*CONDITIONS, "both"],
                        help="with-history runs the whole conversation; "
                             "no-history runs the follow-up on its own")
    parser.add_argument("--loops", default="repair", choices=["repair", "all", "none"],
                        help="which quality loops to compile; repair is what ships")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", action="store_true",
                        help="also ask a model whether the rewrite matches the "
                             "labelled one")
    args = parser.parse_args()

    with open(DATASET) as f:
        conversations = json.load(f)
    if args.limit:
        conversations = conversations[:args.limit]

    conditions = list(CONDITIONS) if args.condition == "both" else [args.condition]

    from src.graph.builder import GraphAgent
    loops = False if args.loops == "none" else args.loops
    agent = GraphAgent(loops=loops, memory=True)

    conn = sqlite3.connect(DB)
    rows = []
    with CallCounter() as counter:
        for condition in conditions:
            print(f"\n=== {condition} ===")
            for conv in conversations:
                turns = len(conv["turns"]) if condition == "with-history" else 1
                print(f"  {conv['id']} ({turns} turn{'s' if turns > 1 else ''}) "
                      f"{conv['turns'][-1]['question'][:56]}")
                try:
                    rows.append(run_conversation(agent, conv, condition, counter, conn))
                except Exception as e:
                    print(f"       failed: {e}")
    conn.close()

    if args.judge:
        print("\nJudging rewrites against the labelled ones...")
        judge_equivalence(rows)

    summaries = {c: summarise([r for r in rows if r["condition"] == c])
                 for c in conditions}
    report = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "loops": args.loops,
        "conversations": len(conversations),
        "summaries": summaries,
        "rows": rows,
    }

    path = os.path.join(_ROOT, "data", "processed",
                        f"multiturn_report_{report['timestamp']}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print_report(report)
    print(f"\nReport saved to: {path}")


if __name__ == "__main__":
    main()
