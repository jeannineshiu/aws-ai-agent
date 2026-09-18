# src/graph/reconcile.py
"""Find the places where the two specialists contradict each other.

The synthesis prompt already asks the model to say so "if the two specialists
disagree". That is the same bargain the citation placeholders were: a prompt
asking the writer to notice something while it writes. It notices sometimes.
The failure it misses is the expensive one — documentation says a limit is 25
and the query result says 40, and the merged answer picks one, silently, with a
citation attached to whichever half it kept. The reader cannot tell that the
other half ever existed.

So the comparison is its own call, before the merge, with structured output. It
returns the contradictions as a list rather than a verdict, for two reasons: the
merge prompt can be handed the specific claims instead of being asked to find
them again, and the caller gets something it can show the user beside the answer.

This is not the critic. The critic asks whether a claim has *any* support in the
sources; a claim supported by one specialist and denied by the other passes that
check, because the support is real. Contradiction between two supported claims
is a different question and needs its own one.

Only the `both` route can produce a conflict, so this only ever runs there.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Reconciliation(BaseModel):
    conflicts: list[str] = Field(
        default_factory=list,
        description=(
            "One entry per contradiction, each naming the fact and both claims. "
            "Empty when the two answers are compatible."
        ),
    )


DETECT_PROMPT = ChatPromptTemplate.from_template("""
Two specialists answered one question from different sources. The documentation
specialist read AWS documentation; the data specialist queried a database of
GitHub issues and Stack Overflow questions. Find the points where they
contradict each other.

Question: {question}

Documentation specialist:
{rag_answer}

Data specialist:
{sql_answer}
{evidence}
A contradiction is the same fact asserted two incompatible ways: different
numbers for one quantity, opposite statements about whether something is
supported, or two different answers to a "which one" question.

These are NOT contradictions:
- Each specialist covering something the other is silent on. Two halves of an
  answer are the normal case, not a disagreement.
- Different levels of detail, or the same fact in different words.
- Documentation describing a capability while the data shows developers
  struggling with it. Both can be true at once.
- A figure from the query result that no document mentions. The database is
  evidence in its own right, not a claim about the documentation.
- Counts measured over different things, periods or scopes.

For each contradiction, write one sentence naming the disputed fact and what
each specialist said about it. Report only what is actually stated — do not
infer a disagreement from what a specialist left out.

Return an empty list if they are compatible.
""")


def render_evidence(sql_finding: dict) -> str:
    """The query result behind the data answer, when there is one.

    The prose from the SQL pipeline is a summary of a DataFrame, and a summary
    can round, truncate or pick a row. Where the two answers disagree about a
    number, the rows are what settles whether the disagreement is real.
    """
    data = sql_finding.get("data")
    if data is None:
        return ""
    try:
        rendered = data.to_string(index=False)
    except AttributeError:
        rendered = str(data)
    if not rendered.strip():
        return ""
    return f"\nThe data specialist's query returned:\n{rendered[:1200]}\n"


class ConflictDetector:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)
        self._detector = self.llm.with_structured_output(Reconciliation)

    def detect(self, question: str, rag_finding: dict, sql_finding: dict) -> list[str]:
        """The contradictions between two findings, or [] if there are none.

        Compatible by default, on the critic's reasoning: a detector that
        reported a conflict whenever its own call failed would put a warning on
        answers nobody disagreed about, and warnings that fire on nothing are
        read as noise within a day.
        """
        rag_answer = (rag_finding.get("answer") or "").strip()
        sql_answer = (sql_finding.get("answer") or "").strip()
        if not rag_answer or not sql_answer:
            return []
        try:
            out = self._detector.invoke(DETECT_PROMPT.format_messages(
                question=question,
                rag_answer=rag_answer[:4000],
                sql_answer=sql_answer[:4000],
                evidence=render_evidence(sql_finding),
            ))
            if out is None:
                return []
            # A model that has nothing to report sometimes says so in prose
            # rather than by returning nothing at all.
            return [c.strip() for c in out.conflicts
                    if c and c.strip() and c.strip().lower() not in _NOTHING]
        except Exception:
            return []


_NOTHING = {"none", "no conflicts", "no contradictions", "n/a", "no conflict"}
