# scripts/test_agent.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import AWSAgent


def print_result(result: dict):
    print(f"\nQ: {result['question']}")
    print(f"Route: {result['route']}")
    print(f"Answer:\n{result['answer'][:500]}")

    if result.get("sql"):
        print(f"SQL: {result['sql']}")

    if result.get("data") is not None:
        print(f"Data:\n{result['data'].to_string(index=False)}")

    if result.get("citations"):
        titles = [c["title"][:40] for c in result["citations"]]
        print(f"Citations: {titles}")

    print("-" * 60)


def main():
    agent = AWSAgent()

    questions = [
        # Should route to RAG
        "What is Amazon Bedrock and what can it do?",
        # Should route to SQL
        "Which year had the most SageMaker questions on Stack Overflow?",
        # Should route to BOTH
        "What are the most common SageMaker issues and how does SageMaker training work?",
    ]

    for q in questions:
        result = agent.run(q)
        print_result(result)


if __name__ == "__main__":
    main()