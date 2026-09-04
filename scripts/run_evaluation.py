# scripts/run_evaluation.py
"""End-to-end evaluation, through the agent rather than around it.

The earlier version called RAGPipeline and SQLPipeline directly. That measured
retrieval and text-to-SQL, which is worth measuring, but it meant routing,
dispatch and every loop were invisible to the harness — the `both` route carried
a correctness bug for the whole life of the project without a single number
touching it.

This runs whichever implementation is asked for and scores what came out:

    python scripts/run_evaluation.py --version v1        # AWSAgent
    python scripts/run_evaluation.py --version v2        # graph, loops on
    python scripts/run_evaluation.py --version v2-flat   # graph, loops off

Report JSON keeps the rag_metrics / sql_metrics / sql_details keys the
Streamlit dashboard reads, and adds agent_metrics and cost_metrics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import sqlite3
import time
from datetime import datetime

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(_ROOT, "data", "processed", "issues.db")


# ── counting what a run costs ─────────────────────────────────────────────────

class CallCounter:
    """Wraps the LLM and embedding entry points to count calls per question.

    Agent quality is only half the comparison; a version that is better because
    it makes three times as many calls should have to say so.
    """

    def __init__(self):
        self.llm = self.embed = self.tokens = 0
        self._patches = []

    def __enter__(self):
        from langchain_openai import ChatOpenAI
        from langchain_openai.embeddings import OpenAIEmbeddings

        for cls, attr, field in ((ChatOpenAI, "invoke", "llm"),
                                 (OpenAIEmbeddings, "embed_query", "embed")):
            original = getattr(cls, attr)
            self._patches.append((cls, attr, original))

            def wrapper(inner_self, *args, _o=original, _f=field, **kwargs):
                setattr(self, _f, getattr(self, _f) + 1)
                out = _o(inner_self, *args, **kwargs)
                # Calls are the coarse cost; tokens are the one that moves when
                # the prompt grows without the call count changing - carrying a
                # conversation forward, for one.
                usage = getattr(out, "usage_metadata", None) or {}
                self.tokens += usage.get("total_tokens", 0)
                return out

            setattr(cls, attr, wrapper)
        return self

    def __exit__(self, *exc):
        for cls, attr, original in self._patches:
            setattr(cls, attr, original)

    def reset(self):
        self.llm = self.embed = self.tokens = 0


# ── the implementations under test ────────────────────────────────────────────

# Which loops each preset switches on. Phase 4 exists to decide this, so it has
# to be a knob rather than a constant.
PRESETS = {
    "v2":        dict(grader=True,  repairer=True,  critic=True),
    "v2-flat":   dict(grader=False, repairer=False, critic=False),
    "v2-repair": dict(grader=False, repairer=True,  critic=False),
    "v2-nograde":dict(grader=False, repairer=True,  critic=True),
}


def build_agent(version: str):
    if version == "v1":
        from src.agent.agent import AWSAgent
        return AWSAgent(), False

    from src.graph.builder import GraphAgent
    from src.graph.critic import Critic
    from src.graph.grader import RetrievalGrader
    from src.graph.repair import SQLRepairer

    on = PRESETS[version]
    return GraphAgent(
        grader=RetrievalGrader() if on["grader"] else None,
        repairer=SQLRepairer() if on["repairer"] else None,
        critic=Critic() if on["critic"] else None,
        loops=False,          # presets decide; do not let the facade fill in
    ), True


def run_one(agent, question: str, traced: bool) -> dict:
    """Normalise both implementations to one shape."""
    if traced:
        state = agent.run_traced(question)
        findings = state.get("findings", [])
        return {
            "answer": state.get("answer", ""),
            "route": state.get("route", "rag"),
            "data": state.get("data"),
            "sql": state.get("sql"),
            "contexts": [t for f in findings for t in (f.get("retrieved_texts") or [])],
            "agents": [f["agent"] for f in findings],
            "trajectory": state.get("trajectory", []),
            "elapsed": state.get("elapsed", 0.0),
            "findings": findings,
        }

    t0 = time.perf_counter()
    out = agent.run(question)
    return {
        "answer": out.get("answer", ""),
        "route": out.get("route", "rag"),
        "data": out.get("data"),
        "sql": out.get("sql"),
        # v1 discards retrieved text, so re-derive it the only way available.
        "contexts": list(getattr(agent, "_last_contexts", []) or []),
        "agents": {"rag": ["rag"], "sql": ["sql"], "both": ["rag", "sql"]}.get(
            out.get("route"), []),
        "trajectory": [],
        "elapsed": time.perf_counter() - t0,
        "findings": [],
    }


def instrument_v1_contexts(agent):
    """v1 throws retrieved chunks away, but RAGAS needs them.

    Wrap RAGPipeline.run to keep the last set. This changes nothing about what
    v1 answers — without it v1 simply cannot be scored on context metrics.
    """
    original = agent.rag.run
    agent._last_contexts = []

    def wrapped(query):
        result = original(query)
        agent._last_contexts = result.get("retrieved_texts", [])
        return result

    agent.rag.run = wrapped
    return agent


# ── scoring ───────────────────────────────────────────────────────────────────

def first_value(df):
    return None if df is None or len(df) == 0 else df.iloc[0, 0]


def values_match(got, want) -> bool:
    """Strings equal ignoring case and surrounding space, numbers to 5%.

    The original harness compared strings exactly, and that cost two correct
    answers in the Phase 5 run: a query that returns 'SageMaker' where the
    labelled value is 'sagemaker' has ranked the services correctly. The
    labels are hand-written and their case is arbitrary, so exact comparison
    was measuring the label, not the answer.

    Both sides of every published comparison are recomputed under this rule
    from the stored `sql_details` of the earlier reports, so relaxing it does
    not silently improve a number by changing what was counted.
    """
    if got is None or want is None:
        return False
    if isinstance(want, str):
        return str(got).strip().casefold() == want.strip().casefold()
    try:
        return abs(float(got) - float(want)) <= max(1, abs(float(want)) * 0.05)
    except (TypeError, ValueError):
        return False


def score_rag(records: list[dict]) -> dict:
    """RAGAS over every sample that produced retrieved context."""
    usable = [r for r in records if r["contexts"] and r["sample"].get("ground_truth")]
    if not usable:
        return {k: float("nan") for k in
                ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}

    from openai import OpenAI
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.llms import llm_factory
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings as LCEmbeddings

    dataset = Dataset.from_dict({
        "question":     [r["sample"]["question"] for r in usable],
        "answer":       [r["answer"] for r in usable],
        "contexts":     [r["contexts"] for r in usable],
        "ground_truth": [r["sample"]["ground_truth"] for r in usable],
    })

    llm = llm_factory("gpt-4o-mini", client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                      max_tokens=4096)
    embeddings = LangchainEmbeddingsWrapper(LCEmbeddings(model="text-embedding-3-small"))

    scores = evaluate(dataset=dataset, metrics=[
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ])
    return {k: mean_score(scores[k]) for k in
            ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}


def mean_score(value) -> float:
    """RAGAS 0.4.x returns per-sample lists; take the mean, ignoring NaN/None."""
    if isinstance(value, (list, tuple)):
        clean = [v for v in value
                 if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(clean) / len(clean) if clean else float("nan")
    return float("nan") if value is None else float(value)


def score_sql(records: list[dict], conn) -> pd.DataFrame:
    rows = []
    for r in records:
        sample = r["sample"]
        if not sample.get("ground_truth_sql"):
            continue
        try:
            expected = first_value(pd.read_sql(sample["ground_truth_sql"], conn))
        except Exception as e:
            print(f"    ground-truth SQL failed: {e}")
            continue
        got = first_value(r["data"])
        rows.append({
            "question": sample["question"],
            "type": sample["type"],
            "adversarial": bool(sample.get("adversarial")),
            "generated_sql": r["sql"] or "",
            "ground_truth_sql": sample["ground_truth_sql"],
            "generated_value": got,
            "ground_truth_value": expected,
            "results_match": values_match(got, expected),
        })
    return pd.DataFrame(rows)


def score_agent(records: list[dict]) -> dict:
    """Delegation and the loops — none of which the previous harness could see."""
    labelled = [r for r in records if r["sample"].get("expected_agents")]
    exact = order = 0
    for r in labelled:
        expected = r["sample"]["expected_agents"]
        got = r["agents"]
        if set(got) == set(expected):
            exact += 1
            if r["sample"].get("expected_order") != "sequential" or got == expected:
                order += 1

    loops = {"rag_retry": 0, "sql_repair": 0, "critic_redraft": 0}
    for r in records:
        traj = r["trajectory"]
        loops["rag_retry"] += sum(
            1 for a, b in zip(traj, traj[1:]) if a == "rag" and b == "rag")
        loops["sql_repair"] += sum(
            1 for a, b in zip(traj, traj[1:]) if a == "sql" and b == "sql")
        loops["critic_redraft"] += sum(
            1 for a, b in zip(traj, traj[1:]) if a == "critic" and b == "synthesize")

    n = len(labelled) or 1
    return {
        "delegation_accuracy": round(exact / n, 3),
        "order_accuracy": round(order / n, 3),
        "labelled_samples": len(labelled),
        "loops_fired": loops,
    }


# ── report ────────────────────────────────────────────────────────────────────

def build_report(version, rag_metrics, sql_df, agent_metrics, records) -> dict:
    accuracy = sql_df["results_match"].mean() if not sql_df.empty else 0.0
    adversarial = sql_df[sql_df["adversarial"]] if not sql_df.empty else pd.DataFrame()

    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "version": version,
        "rag_metrics": {k: (None if math.isnan(v) else round(v, 3))
                        for k, v in rag_metrics.items()},
        "sql_metrics": {
            "accuracy": round(float(accuracy), 3),
            "total_queries": len(sql_df),
            "correct_queries": int(sql_df["results_match"].sum()) if not sql_df.empty else 0,
            "adversarial_accuracy": (round(float(adversarial["results_match"].mean()), 3)
                                     if not adversarial.empty else None),
            "adversarial_total": len(adversarial),
        },
        "agent_metrics": agent_metrics,
        "cost_metrics": {
            "llm_calls_total": sum(r["llm_calls"] for r in records),
            "llm_calls_per_query": round(
                sum(r["llm_calls"] for r in records) / max(len(records), 1), 2),
            "embed_calls_total": sum(r["embed_calls"] for r in records),
            "tokens_total": sum(r.get("tokens", 0) for r in records),
            "tokens_per_query": round(
                sum(r.get("tokens", 0) for r in records) / max(len(records), 1)),
            "seconds_total": round(sum(r["elapsed"] for r in records), 1),
            "seconds_p50": round(sorted(r["elapsed"] for r in records)[len(records) // 2], 2)
            if records else 0.0,
        },
        "sql_details": sql_df.to_dict(orient="records"),
    }


def print_report(report: dict):
    print("\n" + "=" * 68)
    print(f"EVALUATION REPORT — {report['version']}")
    print("=" * 68)

    print("\n--- RAG (RAGAS) ---")
    for metric, score in report["rag_metrics"].items():
        if score is None:
            print(f"  {metric:<22} n/a")
        else:
            print(f"  {metric:<22} {score:.3f}  {'#' * int(score * 20)}")

    m = report["sql_metrics"]
    print("\n--- SQL ---")
    print(f"  accuracy               {m['accuracy']:.1%}  "
          f"({m['correct_queries']}/{m['total_queries']})")
    if m["adversarial_total"]:
        aa = m["adversarial_accuracy"]
        print(f"  adversarial subset     {aa:.1%}  ({m['adversarial_total']} samples)")

    a = report["agent_metrics"]
    print("\n--- Agent ---")
    print(f"  delegation accuracy    {a['delegation_accuracy']:.1%}  "
          f"({a['labelled_samples']} labelled)")
    print(f"  order accuracy         {a['order_accuracy']:.1%}")
    fired = a["loops_fired"]
    print(f"  loops fired            retrieval {fired['rag_retry']}  "
          f"sql repair {fired['sql_repair']}  redraft {fired['critic_redraft']}")

    c = report["cost_metrics"]
    print("\n--- Cost ---")
    print(f"  LLM calls              {c['llm_calls_total']} "
          f"({c['llm_calls_per_query']} per query)")
    print(f"  LLM tokens             {c['tokens_total']} "
          f"({c['tokens_per_query']} per query)")
    print(f"  wall clock             {c['seconds_total']}s total, "
          f"{c['seconds_p50']}s p50")

    failures = [r for r in report["sql_details"] if not r["results_match"]]
    if failures:
        print("\n--- SQL failures ---")
        for row in failures:
            tag = "adv" if row["adversarial"] else row["type"]
            print(f"  [{tag}] {row['question'][:52]}")
            print(f"        expected {row['ground_truth_value']}  "
                  f"got {row['generated_value']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2",
                        choices=["v1", "v2", "v2-flat", "v2-repair", "v2-nograde"],
                        help="v1=AWSAgent; v2=all loops; v2-flat=none; "
                             "v2-repair=SQL repair only; v2-nograde=repair+critic")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N samples")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="agent and SQL metrics only; no RAGAS judge calls")
    args = parser.parse_args()

    with open(os.path.join(_ROOT, "data", "processed", "ground_truth.json")) as f:
        samples = json.load(f)
    if args.limit:
        samples = samples[:args.limit]

    agent, traced = build_agent(args.version)
    if not traced:
        agent = instrument_v1_contexts(agent)

    print(f"\nRunning {len(samples)} samples through {args.version}...")
    records = []
    with CallCounter() as counter:
        for i, sample in enumerate(samples, 1):
            print(f"  [{i:>2}/{len(samples)}] {sample['type']:<5} {sample['question'][:58]}")
            counter.reset()
            try:
                result = run_one(agent, sample["question"], traced)
            except Exception as e:
                print(f"       failed: {e}")
                result = {"answer": "", "route": "rag", "data": None, "sql": None,
                          "contexts": [], "agents": [], "trajectory": [], "elapsed": 0.0}
            result["sample"] = sample
            result["llm_calls"], result["embed_calls"] = counter.llm, counter.embed
            result["tokens"] = counter.tokens
            records.append(result)

    conn = sqlite3.connect(DB)
    sql_df = score_sql(records, conn)
    conn.close()

    agent_metrics = score_agent(records)
    rag_metrics = ({k: float("nan") for k in
                    ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
                   if args.skip_ragas else score_rag(records))

    report = build_report(args.version, rag_metrics, sql_df, agent_metrics, records)
    path = os.path.join(_ROOT, "data", "processed",
                        f"eval_report_{report['timestamp']}_{args.version}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print_report(report)
    print(f"\nReport saved to: {path}")


if __name__ == "__main__":
    main()
