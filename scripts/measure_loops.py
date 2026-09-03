# scripts/measure_loops.py
"""Measure what the Phase 2 loops are worth, on the ground-truth SQL set.

run_evaluation.py cannot answer this: it calls SQLPipeline directly and never
enters the graph, so the repair loop is invisible to it. Wiring the harness to
run either implementation is Phase 4. This is the narrower question that can be
answered now — does the repair loop recover queries v1 gets wrong?

    python scripts/measure_loops.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3

import pandas as pd

from src.graph.builder import GraphAgent
from src.graph.supervisor import Plan
from src.sql.pipeline import SQLPipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SQLOnly:
    """Pins dispatch to the SQL specialist so the comparison isolates the loop."""
    def plan(self, question): return Plan(agents=["sql"], mode="parallel")
    def refine(self, question, finding, next_agent): return question


def matches(got, want) -> bool:
    """The same comparison run_evaluation.py uses: strings exact, numbers to 5%."""
    if got is None or want is None:
        return False
    if isinstance(want, str):
        return str(got).strip() == want.strip()
    try:
        tolerance = max(1, abs(float(want)) * 0.05)
        return abs(float(got) - float(want)) <= tolerance
    except (TypeError, ValueError):
        return False


def first_value(df):
    return None if df is None or len(df) == 0 else df.iloc[0, 0]


def main():
    with open(os.path.join(_ROOT, "data", "processed", "ground_truth.json")) as f:
        samples = [s for s in json.load(f) if s["type"] == "sql"]

    conn = sqlite3.connect(os.path.join(_ROOT, "data", "processed", "issues.db"))
    v1 = SQLPipeline()
    v2 = GraphAgent(supervisor=SQLOnly())

    rows = []
    for s in samples:
        q, gt_sql = s["question"], s["ground_truth_sql"]
        try:
            expected = first_value(pd.read_sql(gt_sql, conn))
        except Exception as e:
            print(f"  ground-truth SQL failed: {e}")
            continue

        a = v1.run(q)
        b = v2.run(q)

        rows.append({
            "question": q,
            "expected": expected,
            "v1": first_value(a.get("data")),
            "v2": first_value(b.get("data")),
            "v1_ok": matches(first_value(a.get("data")), expected),
            "v2_ok": matches(first_value(b.get("data")), expected),
        })

    conn.close()

    print("\n" + "=" * 82)
    print("SQL REPAIR LOOP — v1 vs v2 on the ground-truth set")
    print("=" * 82)
    print(f"{'':7s} {'question':<46s} {'expected':>10s} {'v1':>8s} {'v2':>8s}")
    for r in rows:
        mark = f"{'OK' if r['v1_ok'] else '--'}/{'OK' if r['v2_ok'] else '--'}"
        print(f"{mark:7s} {r['question'][:46]:<46s} "
              f"{str(r['expected'])[:10]:>10s} {str(r['v1'])[:8]:>8s} {str(r['v2'])[:8]:>8s}")

    n = len(rows)
    v1_ok = sum(r["v1_ok"] for r in rows)
    v2_ok = sum(r["v2_ok"] for r in rows)
    recovered = sum(1 for r in rows if r["v2_ok"] and not r["v1_ok"])
    broken = sum(1 for r in rows if r["v1_ok"] and not r["v2_ok"])

    print("-" * 82)
    print(f"   v1 {v1_ok}/{n}   v2 {v2_ok}/{n}   recovered {recovered}   regressed {broken}")
    if v1_ok < n:
        print(f"   recovery rate: {recovered}/{n - v1_ok} of v1's failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
