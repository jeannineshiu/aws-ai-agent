# src/graph/supervisor.py
"""The supervisor: decides which specialists run, and in what relationship.

This replaces QueryRouter's single classification. Two things change.

First, the decision is structured rather than parsed from free text.
QueryRouter mapped any unrecognised reply to RAG through
`route_map.get(raw, RouteType.RAG)`, which silently sends statistics questions
to a pipeline that cannot count. `with_structured_output` makes the model
return a validated object instead, so that fallback has no work to do.

Second, the supervisor decides *ordering*, not just membership. Some questions
need both specialists working independently; others need the second specialist
to see the first one's answer before it can even form its query. v1 could not
express the difference, which is why the `both` route sent the raw question to
the retriever and got back nothing relevant.
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

AGENTS = ("rag", "sql")


class Plan(BaseModel):
    """Which specialists to run, and how they relate."""
    agents: list[Literal["rag", "sql"]] = Field(
        description="Specialists needed. For sequential mode, list the one that must run first."
    )
    mode: Literal["parallel", "sequential"] = Field(
        description="Whether the second specialist depends on the first one's answer."
    )


class ContextualPlan(Plan):
    """A plan for a follow-up question, plus the question with its references resolved.

    Kept as a separate schema rather than an optional field on `Plan` so the
    first turn calls the model with exactly the schema it used before Phase 3.
    Adding a field would have changed every single-turn decision the evaluation
    measured, to buy something a first turn cannot use.
    """
    standalone_question: str = Field(
        description="The question rewritten to stand on its own, with every "
                    "reference to the conversation replaced by what it refers to."
    )


class Refinement(BaseModel):
    """The follow-up question for the second specialist in a sequential plan."""
    query: str = Field(description="A single self-contained question.")


PLAN_PROMPT = ChatPromptTemplate.from_template("""
You are the supervisor of an AWS AI/ML assistant. Two specialists are available.

- rag : AWS documentation. Explains how services work, their concepts, features,
        comparisons and best practices. It cannot count or rank anything.
- sql : A database of GitHub issues and Stack Overflow questions. Produces counts,
        rankings, trends and aggregates. It cannot explain how a service works.

Include a specialist only if the question cannot be answered without it. Adding
one that is not needed doubles the cost and dilutes the answer.

mode (ignored when only one specialist is chosen):
- "parallel"   : both specialists can work from the original question, because
                 each half names what it is about.
- "sequential" : the second specialist cannot form its query until the first has
                 answered, because the question identifies something by a
                 statistic and the identity is unknown until the data is queried.
                 List the specialist that resolves the unknown first - this is
                 almost always sql, since it is the one that ranks and counts.

Examples:

  "What is Amazon Bedrock?"
    agents ["rag"] - documentation only, nothing to count.

  "How many SageMaker questions were asked in 2023?"
    agents ["sql"] - a count, and the service is already named.

  "Which AWS service has the most unanswered questions?"
    agents ["sql"] - a ranking. The answer is a name, not an explanation.

  "What are the most common SageMaker issues, and how does training work?"
    agents ["sql", "rag"], mode "parallel" - the service is named in the question,
    so the documentation half can be searched without waiting for the data half.

  "Which service has the most questions and what does it do?"
    agents ["sql", "rag"], mode "sequential" - which service is unknown until the
    database is queried, and only then can the documentation be searched for it.

Question: {question}
""")

# The follow-up prompt is the planning prompt with the conversation prepended and
# one extra job. Resolution and dispatch happen in the same structured call: the
# specialists cannot be chosen without resolving the reference first, so asking
# twice would pay for the same reasoning twice.
FOLLOWUP_PROMPT = ChatPromptTemplate.from_template("""
Earlier in this conversation:
{history}

The user now asks: {question}

First rewrite that into a question that stands on its own. Replace every
pronoun, "it", "that service", "the same thing", and every omitted subject with
the concrete thing it refers to in the conversation above. If the question
already stands on its own, repeat it unchanged - a follow-up is not always
about what came before.

Then plan the dispatch for the rewritten question, not for the words the user
typed.

Two specialists are available.

- rag : AWS documentation. Explains how services work, their concepts, features,
        comparisons and best practices. It cannot count or rank anything.
- sql : A database of GitHub issues and Stack Overflow questions. Produces counts,
        rankings, trends and aggregates. It cannot explain how a service works.

Include a specialist only if the question cannot be answered without it. Adding
one that is not needed doubles the cost and dilutes the answer.

mode (ignored when only one specialist is chosen):
- "parallel"   : both specialists can work from the rewritten question, because
                 each half names what it is about.
- "sequential" : the second specialist cannot form its query until the first has
                 answered. List the specialist that resolves the unknown first -
                 almost always sql, since it is the one that ranks and counts.

Examples, given "What is Amazon Bedrock?" earlier in the conversation:

  "How much does it cost?"
    standalone "How much does Amazon Bedrock cost?", agents ["rag"].

  "How many questions are there about it?"
    standalone "How many questions are there about Amazon Bedrock?", agents ["sql"].

  "What is Amazon Comprehend?"
    standalone "What is Amazon Comprehend?", agents ["rag"] - a new subject,
    nothing to resolve.
""")

REFINE_PROMPT = ChatPromptTemplate.from_template("""
You are the supervisor of an AWS AI/ML assistant.

Original question: {question}

The {done_agent} specialist has answered:
{finding}

Write the single question the {next_agent} specialist should now answer, so that
the two answers together address the original question.

Replace anything that depended on the first result with the concrete value it
returned. For example, "what does it do" becomes "what does <the actual service>
do". The question must stand on its own without the original for context.

Return only the question.
""")


def _render_history(history: list[dict], max_turns: int = 4, max_answer: int = 400) -> str:
    """The last few turns, answers truncated.

    Bounded on purpose: the reference a follow-up needs is almost always in the
    turn just before it, and an unbounded transcript would grow the planning
    prompt without bound for the length of a session.
    """
    lines = []
    for turn in history[-max_turns:]:
        lines.append(f"User: {turn.get('question', '')}")
        lines.append(f"Assistant: {(turn.get('answer') or '')[:max_answer]}")
    return "\n".join(lines)


class Supervisor:
    def __init__(self, llm=None):
        # Same cheap model as the router it replaces; planning is a small task.
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)
        self._planner = self.llm.with_structured_output(Plan)
        self._followup_planner = self.llm.with_structured_output(ContextualPlan)
        self._refiner = self.llm.with_structured_output(Refinement)

    def plan(self, question: str, history: list[dict] | None = None) -> Plan:
        """Decide the dispatch. Falls back to rag-only if the model returns nothing usable.

        With earlier turns to draw on, the same call also resolves what the
        question refers to; `standalone_question` carries the result. Without
        them the call is byte-identical to the one Phase 1 made, so the
        single-turn numbers stay comparable.
        """
        try:
            if history:
                plan = self._followup_planner.invoke(FOLLOWUP_PROMPT.format_messages(
                    history=_render_history(history), question=question))
            else:
                plan = self._planner.invoke(PLAN_PROMPT.format_messages(question=question))
        except Exception:
            plan = None

        agents = []
        if plan is not None:
            # Preserve order, drop duplicates and anything outside the vocabulary.
            for a in plan.agents:
                if a in AGENTS and a not in agents:
                    agents.append(a)

        # An empty or unusable rewrite degrades to the words the user typed,
        # which is exactly the pre-Phase-3 behaviour.
        standalone = (getattr(plan, "standalone_question", "") or "").strip() or question

        if not agents:
            # RAG is the safer default: it says it lacks the information rather
            # than inventing a query against the database.
            return ContextualPlan(agents=["rag"], mode="parallel",
                                  standalone_question=standalone)

        return ContextualPlan(agents=agents, mode=plan.mode if plan else "parallel",
                              standalone_question=standalone)

    def refine(self, question: str, finding: dict, next_agent: str) -> str:
        """Rewrite the question for the next specialist using what the first returned.

        This is the edge v1 did not have. Without it the retriever searches the
        original phrasing - "which service has the most questions and what does it
        do" - which matches nothing in the documentation.
        """
        try:
            out = self._refiner.invoke(REFINE_PROMPT.format_messages(
                question=question,
                done_agent=finding.get("agent", "other"),
                finding=(finding.get("answer") or "")[:1500],
                next_agent=next_agent,
            ))
            if out and out.query.strip():
                return out.query.strip()
        except Exception:
            pass
        return question   # degrade to v1 behaviour rather than failing the turn
