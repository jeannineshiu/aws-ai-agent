# src/graph/critic.py
"""Check that the answer is supported by what was actually retrieved.

This is RAGAS Faithfulness moved from the offline harness into the request
path. The offline metric tells you last night's answers were 0.94 grounded;
this one can still do something about tonight's.

It only runs where there is retrieved evidence to check against. A SQL-only
answer is grounded in a DataFrame the pipeline computed deterministically, so
there is no hallucination surface worth an LLM call.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Verdict(BaseModel):
    grounded: bool = Field(
        description="True if every substantive claim is supported by the context."
    )
    unsupported: str = Field(
        default="",
        description="The claims that lack support, when grounded is False.",
    )


CRITIC_PROMPT = ChatPromptTemplate.from_template("""
You are checking an AWS AI/ML assistant's answer against its sources.

Question: {question}

Source material the answer was written from:
{context}

The answer:
{answer}

Is every substantive claim in the answer supported by the source material?

Judge support, not style or completeness. Specifically:
- The source material includes query results as well as documents. A figure that
  appears in a query result is supported, even though no document mentions it.
- An answer that says the sources do not cover something is grounded. Admitting
  a gap is correct behaviour, not a failure.
- Ignore citation markers and formatting.
- Numbers, service names and capabilities that appear nowhere in the sources are
  unsupported, however plausible they sound.

If unsupported, name the claims.
""")


class Critic:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)
        self._critic = self.llm.with_structured_output(Verdict)

    def check(self, question: str, answer: str, contexts: list[str]) -> Verdict:
        """Grounded by default. A critic that fails closed on its own errors
        would reject good answers and spend the revision budget doing it."""
        if not contexts or not answer.strip():
            return Verdict(grounded=True)
        try:
            joined = "\n\n---\n\n".join(c[:1200] for c in contexts)[:8000]
            out = self._critic.invoke(CRITIC_PROMPT.format_messages(
                question=question, context=joined, answer=answer[:4000]))
            if out is not None:
                return out
        except Exception:
            pass
        return Verdict(grounded=True)
