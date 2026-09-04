# scripts/compare_v1_v2.py
"""Phase 0 parity check against the real stack.

tests/test_graph.py already proves v1 and v2 return identical dicts given
identical pipeline output. This script covers what fakes cannot: that the graph
wires up correctly against the real Chroma index, SQLite database and OpenAI
client, and routes the same way v1 does.

Answer text is generated and will differ between runs, so the comparison is
structural — route taken, which specialists ran, which fields are populated.

    python scripts/compare_v1_v2.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from src.agent.agent import AWSAgent
from src.graph.builder import GraphAgent

# The project's own canonical examples, taken from app.py's sidebar.
QUESTIONS = [
    "What is Amazon Bedrock?",
    "How does SageMaker Model Monitor work?",
    "When should I use Rekognition vs Comprehend?",
    "Which AWS service has the most unanswered questions?",
    "How many SageMaker questions were asked in 2023?",
    "Which repo has the most open GitHub issues?",
    "What are the most common SageMaker issues and how does training work?",
    "Which service has the most questions and what does it do?",
]


def shape(result: dict) -> dict:
    """The parts of a result that must not change when only the machinery changes."""
    return {
        "route": result["route"],
        "has_answer": bool(result["answer"].strip()),
        "citations": len(result["citations"]),
        "has_data": result["data"] is not None,
        "has_sql": result["sql"] is not None,
    }


def main():
    print("Building v1 (AWSAgent)...")
    v1 = AWSAgent()
    print("Building v2 (GraphAgent)...")
    v2 = GraphAgent()

    rows = []
    for q in QUESTIONS:
        print(f"\n── {q}")
        t0 = time.perf_counter(); a = shape(v1.run(q)); t1 = time.perf_counter()
        t2 = time.perf_counter(); b = shape(v2.run(q)); t3 = time.perf_counter()
        rows.append((q, a, b, t1 - t0, t3 - t2))

    print("\n" + "=" * 78)
    print("PHASE 0 PARITY")
    print("=" * 78)
    print(f"{'':2s} {'question':<44s} {'v1':>10s} {'v2':>10s}  {'sec':>8s}")

    mismatches = 0
    for q, a, b, ta, tb in rows:
        same = a == b
        mismatches += not same
        mark = "OK" if same else "!!"
        print(f"{mark:2s} {q[:44]:<44s} {a['route']:>10s} {b['route']:>10s}  {ta:4.1f}/{tb:4.1f}")
        if not same:
            for k in a:
                if a[k] != b[k]:
                    print(f"     {k}: v1={a[k]!r}  v2={b[k]!r}")

    v1_total = sum(r[3] for r in rows)
    v2_total = sum(r[4] for r in rows)
    print("-" * 78)
    print(f"   {len(rows) - mismatches}/{len(rows)} identical"
          f"   ·   v1 {v1_total:.1f}s   v2 {v2_total:.1f}s")

    if mismatches:
        print("\nPhase 0 requires structural parity. Investigate before continuing.")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
