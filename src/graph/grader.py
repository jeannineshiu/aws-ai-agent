# src/graph/grader.py
"""Judge whether retrieved documents can answer the question, and rewrite the
query when they cannot.

The failure this exists for was measured, not imagined. Retrieving for
"How does SageMaker Model Monitor work?" returns four chunks and three of them
come from the same page, so the effective evidence is one document wearing four
hats. Nothing downstream notices: the generator writes a confident answer from
whatever it was handed, and the only signal is a Context Recall score computed
hours later, offline.

Grading moves that judgement to where it can still change the outcome.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Grade(BaseModel):
    sufficient: bool = Field(
        description="True if the documents contain enough to answer the question."
    )
    missing: str = Field(
        default="",
        description="What is absent, when sufficient is False. Empty otherwise.",
    )


class Rewrite(BaseModel):
    query: str = Field(description="A reformulated search query.")


GRADE_PROMPT = ChatPromptTemplate.from_template("""
You are grading retrieval quality for an AWS AI/ML assistant.

Question: {question}

Retrieved documents:
{context}

Can these documents answer the question?

Judge coverage, not eloquence. Answer False when:
- the documents are about a different service or topic than the question
- they mention the subject only in passing, without the detail the question asks for
- they are several excerpts of the same passage, so there is really only one source

Answer True when the documents contain the facts needed, even if worded differently
from the question.

If False, say briefly what is missing.
""")

REWRITE_PROMPT = ChatPromptTemplate.from_template("""
A vector search over AWS documentation returned nothing useful.

Original question: {question}
Search query that failed: {query}
What was missing: {missing}

Write a better search query. Vector search matches on wording, so:
- use the terminology AWS documentation uses, not the user's phrasing
- name the service and the specific feature or mechanism
- drop conversational framing and question words
- keep it under fifteen words

Return only the query.
""")


class RetrievalGrader:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)
        self._grader = self.llm.with_structured_output(Grade)
        self._rewriter = self.llm.with_structured_output(Rewrite)

    def grade(self, question: str, docs) -> Grade:
        """Sufficient by default: a grader that fails open degrades to v1
        behaviour, while one that fails closed would burn the retry budget on
        its own errors."""
        if not docs:
            return Grade(sufficient=False, missing="retrieval returned nothing")
        try:
            context = "\n\n---\n\n".join(
                getattr(d, "page_content", str(d))[:1200] for d in docs
            )
            out = self._grader.invoke(GRADE_PROMPT.format_messages(
                question=question, context=context))
            if out is not None:
                return out
        except Exception:
            pass
        return Grade(sufficient=True)

    def rewrite(self, question: str, query: str, missing: str) -> str:
        try:
            out = self._rewriter.invoke(REWRITE_PROMPT.format_messages(
                question=question, query=query, missing=missing or "nothing relevant"))
            if out and out.query.strip():
                return out.query.strip()
        except Exception:
            pass
        return query
