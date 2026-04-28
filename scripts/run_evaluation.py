# scripts/run_evaluation.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import pandas as pd
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings

from src.rag.pipeline import RAGPipeline
from src.sql.pipeline import SQLPipeline


def evaluate_rag(rag_pipeline: RAGPipeline, rag_samples: list[dict]) -> dict:
    """Evaluate RAG pipeline using RAGAS metrics."""
    print(f"\nRunning RAG evaluation on {len(rag_samples)} samples...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, sample in enumerate(rag_samples):
        print(f"  [{i+1}/{len(rag_samples)}] {sample['question'][:60]}")
        result = rag_pipeline.run(sample["question"])

        questions.append(sample["question"])
        answers.append(result["answer"])
        contexts.append(result.get("retrieved_texts") or [""])
        ground_truths.append(sample["ground_truth"])

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Configure RAGAS LLM and embeddings, inject into each metric
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=4096)
    embeddings = LangchainEmbeddingsWrapper(LCOpenAIEmbeddings(model="text-embedding-3-small"))

    scores = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ],
    )

    return scores


def evaluate_sql(sql_pipeline: SQLPipeline, sql_samples: list[dict]) -> pd.DataFrame:
    """Evaluate SQL pipeline by comparing query results to ground truth."""
    print(f"\nRunning SQL evaluation on {len(sql_samples)} samples...")

    conn = sqlite3.connect(os.path.join(_PROJECT_ROOT, "data", "processed", "issues.db"))
    results = []

    for sample in sql_samples:
        question = sample["question"]
        ground_truth_sql = sample["ground_truth_sql"]
        print(f"  Q: {question[:60]}")

        # Run generated SQL
        result = sql_pipeline.run(question)
        generated_sql = result.get("sql", "")
        generated_data = result.get("data")

        # Run ground truth SQL
        try:
            gt_data = pd.read_sql(ground_truth_sql, conn)
        except Exception as e:
            gt_data = None
            print(f"    Ground truth SQL failed: {e}")

        # Compare results
        results_match = False
        if generated_data is not None and gt_data is not None:
            try:
                gen_val = generated_data.iloc[0, 0]
                gt_val = gt_data.iloc[0, 0]

                # String comparison (e.g. tag names)
                if isinstance(gt_val, str):
                    results_match = str(gen_val).strip() == str(gt_val).strip()
                else:
                    # Numeric comparison with 5% tolerance
                    # Accounts for minor SQL generation variations
                    tolerance = max(1, abs(float(gt_val)) * 0.05)
                    results_match = abs(float(gen_val) - float(gt_val)) <= tolerance
            except Exception:
                results_match = False

        results.append({
            "question": question,
            "generated_sql": generated_sql,
            "ground_truth_sql": ground_truth_sql,
            "results_match": results_match,
            "generated_value": generated_data.iloc[0, 0] if generated_data is not None and not generated_data.empty else None,
            "ground_truth_value": gt_data.iloc[0, 0] if gt_data is not None and not gt_data.empty else None,
        })

    conn.close()
    return pd.DataFrame(results)


def _mean_score(val) -> float:
    """RAGAS 0.4.x returns per-sample lists; take the mean, ignoring NaN/None."""
    import math
    if isinstance(val, (list, tuple)):
        vals = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(vals) / len(vals) if vals else float("nan")
    if val is None:
        return float("nan")
    return float(val)


def save_report(rag_scores: dict, sql_df: pd.DataFrame):
    """Save evaluation report to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(_PROJECT_ROOT, "data", "processed", f"eval_report_{timestamp}.json")

    # SQL accuracy
    sql_accuracy = sql_df["results_match"].mean() if not sql_df.empty else 0

    report = {
        "timestamp": timestamp,
        "rag_metrics": {
            "faithfulness": round(_mean_score(rag_scores["faithfulness"]), 3),
            "answer_relevancy": round(_mean_score(rag_scores["answer_relevancy"]), 3),
            "context_precision": round(_mean_score(rag_scores["context_precision"]), 3),
            "context_recall": round(_mean_score(rag_scores["context_recall"]), 3),
        },
        "sql_metrics": {
            "accuracy": round(float(sql_accuracy), 3),
            "total_queries": len(sql_df),
            "correct_queries": int(sql_df["results_match"].sum()),
        },
        "sql_details": sql_df.to_dict(orient="records"),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report, report_path


def print_report(report: dict):
    """Print evaluation report to terminal."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print("\n--- RAG Metrics ---")
    for metric, score in report["rag_metrics"].items():
        if score is None or (isinstance(score, float) and score != score):
            print(f"  {metric:<22} NaN   (evaluation failed)")
        else:
            bar = "█" * int(score * 20)
            print(f"  {metric:<22} {score:.3f}  {bar}")

    print("\n--- SQL Metrics ---")
    m = report["sql_metrics"]
    print(f"  Accuracy:    {m['accuracy']:.1%}  ({m['correct_queries']}/{m['total_queries']} correct)")

    print("\n--- SQL Details ---")
    for row in report["sql_details"]:
        status = "✅" if row["results_match"] else "❌"
        print(f"  {status} {row['question'][:55]}")
        if not row["results_match"]:
            print(f"     Expected: {row['ground_truth_value']}")
            print(f"     Got:      {row['generated_value']}")


def main():
    # Load ground truth
    with open(os.path.join(_PROJECT_ROOT, "data", "processed", "ground_truth.json")) as f:
        all_samples = json.load(f)

    rag_samples = [s for s in all_samples if s["type"] == "rag"]
    sql_samples = [s for s in all_samples if s["type"] == "sql"]

    # Initialize pipelines
    rag = RAGPipeline()
    sql = SQLPipeline()

    # Run evaluations
    rag_scores = evaluate_rag(rag, rag_samples)
    sql_df = evaluate_sql(sql, sql_samples)

    # Save and print report
    report, path = save_report(rag_scores, sql_df)
    print_report(report)
    print(f"\nReport saved to: {path}")


if __name__ == "__main__":
    main()