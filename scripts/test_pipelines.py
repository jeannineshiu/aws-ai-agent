# scripts/test_pipelines.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.pipeline import RAGPipeline
from src.sql.pipeline import SQLPipeline

def test_rag():
    print("=" * 60)
    print("RAG PIPELINE TEST")
    print("=" * 60)

    rag = RAGPipeline()

    questions = [
        "What is Amazon SageMaker and what can it do?",
        "How does Bedrock differ from SageMaker?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = rag.run(q)
        print(f"A: {result['answer'][:400]}...")
        print(f"Sources: {[c['title'][:40] for c in result['citations']]}")


def test_sql():
    print("\n" + "=" * 60)
    print("SQL PIPELINE TEST")
    print("=" * 60)

    sql = SQLPipeline()

    questions = [
        "Which AWS service has the most unanswered questions?",
        "How many SageMaker questions were asked in 2023?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = sql.run(q)
        print(f"SQL: {result['sql']}")
        if result.get("data") is not None:
            print(f"Data:\n{result['data'].to_string(index=False)}")
        print(f"A: {result['answer']}")


if __name__ == "__main__":
    test_rag()
    test_sql()