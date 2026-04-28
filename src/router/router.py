# src/router/router.py
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


class RouteType(str, Enum):
    RAG = "rag"
    SQL = "sql"
    BOTH = "both"


ROUTER_PROMPT = ChatPromptTemplate.from_template("""
You are a query router for an AWS AI/ML assistant.
Classify the user's question into one of three categories:

- "rag"  : Questions about how AWS services work, concepts, features,
           documentation, comparisons, or best practices.
           Examples:
           "How does SageMaker training work?"
           "What is Amazon Bedrock?"
           "How do I deploy a model?"

- "sql"  : Questions about statistics, counts, trends, rankings,
           or analysis of developer questions and GitHub issues.
           Examples:
           "How many SageMaker questions were asked in 2023?"
           "Which service has the most unanswered questions?"
           "What are the most common issues in the SageMaker SDK?"

- "both" : Questions that require BOTH documentation knowledge AND data analysis.
           Examples:
           "Which service has the most questions and how does it work?"
           "What are common Lambda problems and how do I fix them?"

Return ONLY one word: rag, sql, or both.
No explanation, no punctuation.

Question: {question}

Route:""")


class QueryRouter:
    def __init__(self):
        # Use a fast, cheap model for routing
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)

    def route(self, question: str) -> RouteType:
        """Classify a question and return the appropriate route."""
        prompt = ROUTER_PROMPT.format_messages(question=question)
        response = self.llm.invoke(prompt)
        raw = response.content.strip().lower()

        # Map response to RouteType with fallback
        route_map = {
            "rag": RouteType.RAG,
            "sql": RouteType.SQL,
            "both": RouteType.BOTH,
        }
        return route_map.get(raw, RouteType.RAG)