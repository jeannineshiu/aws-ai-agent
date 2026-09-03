# src/graph/synthesizer.py
"""Merge findings from more than one specialist into a single answer.

v1 joined the two answers with an f-string:

    f"**From documentation:**\\n{rag}\\n\\n**From data analysis:**\\n{sql}"

That is two answers printed next to each other, not one answer. The reader is
left to reconcile them, and nothing checks that they are even about the same
thing. This node does the reconciliation with one LLM call.

A single finding is passed through untouched. Synthesising one answer into one
answer costs a call and can only lose fidelity, and it keeps the rag-only and
sql-only routes byte-identical to v1 - which is what lets the existing
evaluation numbers stay comparable.
"""
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template("""
You are an AWS AI/ML assistant. Two specialists investigated one question.
Write the single answer the user should receive.

Question: {question}

Documentation specialist:
{rag_answer}

Data specialist:
{sql_answer}

Rules:
- Lead with what directly answers the question. If the data identified a specific
  service or repository, name it in the first sentence.
- Write one continuous answer. Do not label sections by specialist, and do not
  state the same fact twice.
- Keep every [Source: ...] citation that appears in the documentation answer,
  in that exact format. Drop any citation that is an unfilled template rather
  than a real document title.
- If the two specialists disagree or the documentation does not cover what the
  data found, say so plainly rather than papering over it.
""")


# The RAG prompt instructs the model to always cite. When retrieval turns up
# nothing relevant it complies anyway and emits the format string itself, which
# then survives synthesis as a citation to nothing. The prompt above asks for
# these to be dropped; this strips them regardless, on the same reasoning as
# README section 4 — the prompt reduces how often it happens, the code makes it
# impossible.
_PLACEHOLDER_CITATION = re.compile(
    r"\s*\[Source:\s*(?:[^\]]*<[^\]]*|N/?A\s*\|\s*N/?A\s*)\]", re.IGNORECASE)


REVISION_PROMPT = ChatPromptTemplate.from_template("""
An earlier draft of this answer made claims the sources do not support.

Question: {question}

Source material:
{context}

The draft:
{answer}

Unsupported claims: {critique}

Rewrite the answer so that every statement is traceable to the source material.
Remove or correct the unsupported claims rather than softening them with hedges.
If removing them leaves the question partly unanswered, say plainly which part
the sources do not cover. Keep every real [Source: ...] citation.
""")


class Synthesizer:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60)

    @staticmethod
    def strip_placeholder_citations(text: str) -> str:
        """Remove [Source: ...] markers that still contain <angle-bracket> slots."""
        return _PLACEHOLDER_CITATION.sub("", text).strip()

    def merge(self, question: str, rag_finding: dict, sql_finding: dict) -> str:
        try:
            response = self.llm.invoke(SYNTHESIS_PROMPT.format_messages(
                question=question,
                rag_answer=rag_finding.get("answer", ""),
                sql_answer=sql_finding.get("answer", ""),
            ))
            text = self.strip_placeholder_citations(response.content or "")
            if text:
                return text
        except Exception:
            pass

        # Degrade to v1's concatenation rather than dropping half the work.
        return (
            f"**From documentation:**\n{rag_finding.get('answer','')}\n\n"
            f"**From data analysis:**\n{sql_finding.get('answer','')}"
        )

    def revise(self, question: str, answer: str, contexts: list[str], critique: str) -> str:
        """Redraft an answer the critic rejected, constrained to the sources.

        A groundedness failure is a generation failure, not a retrieval one —
        the evidence was there and the draft went beyond it. So this rewrites
        from the same context rather than searching again.
        """
        try:
            joined = "\n\n---\n\n".join(c[:1200] for c in contexts)[:8000]
            response = self.llm.invoke(REVISION_PROMPT.format_messages(
                question=question, context=joined,
                answer=answer[:4000], critique=critique or "unspecified",
            ))
            text = self.strip_placeholder_citations(response.content or "")
            if text:
                return text
        except Exception:
            pass
        return answer
