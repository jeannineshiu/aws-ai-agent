# src/agent/agent.py
from src.router.router import QueryRouter, RouteType
from src.rag.pipeline import RAGPipeline
from src.sql.pipeline import SQLPipeline


class AWSAgent:
    def __init__(self):
        print("Initializing AWS AI/ML Agent...")
        self.router = QueryRouter()
        self.rag = RAGPipeline()
        self.sql = SQLPipeline()
        print("Agent ready.")

    def run(self, question: str) -> dict:
        """Route question and run appropriate pipeline(s)."""

        # Step 1: Route
        route = self.router.route(question)
        print(f"  → Route: {route.value}")

        # Step 2: Execute
        if route == RouteType.RAG:
            result = self.rag.run(question)
            return {
                "question": question,
                "route": route.value,
                "answer": result["answer"],
                "citations": result.get("citations", []),
                "data": None,
                "sql": None,
            }

        elif route == RouteType.SQL:
            result = self.sql.run(question)
            return {
                "question": question,
                "route": route.value,
                "answer": result["answer"],
                "citations": [],
                "data": result.get("data"),
                "sql": result.get("sql"),
            }

        elif route == RouteType.BOTH:
            print("  → Running RAG...")
            rag_result = self.rag.run(question)
            print("  → Running SQL...")
            sql_result = self.sql.run(question)

            # Combine both answers
            combined_answer = (
                f"**From documentation:**\n{rag_result['answer']}\n\n"
                f"**From data analysis:**\n{sql_result['answer']}"
            )
            return {
                "question": question,
                "route": route.value,
                "answer": combined_answer,
                "citations": rag_result.get("citations", []),
                "data": sql_result.get("data"),
                "sql": sql_result.get("sql"),
            }

        return {
            "question": question,
            "route": "rag",
            "answer": "Unable to process this question. Please try again.",
            "citations": [],
            "data": None,
            "sql": None,
        }