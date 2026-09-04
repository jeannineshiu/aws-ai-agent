# src/graph/supervisor.py
"""The supervisor: decides which specialists run, and in what relationship.

This replaces QueryRouter's single classification. Three things change.

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

Third, the plan says what each specialist is being asked, not just that it was
asked. A two-part question sent whole to both specialists is answered whole by
both: the multi-turn harness caught "how many questions are tagged SageMaker,
and how does training work in it" becoming a count with `AND title LIKE
'%training%'` bolted on - 541 rows instead of 1840, because the documentation
half narrowed the query. Deciding who runs and deciding what they are asked is
one decision, so it is one call.
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

load_dotenv()

AGENTS = ("rag", "sql")


class Task(BaseModel):
    """One specialist, and the question that specialist alone has to answer."""
    agent: Literal["rag", "sql"]
    query: str = Field(
        description="The part of the question this specialist should answer, written "
                    "so it stands on its own. It must carry no clause meant for the "
                    "other specialist."
    )


class Plan(BaseModel):
    """Which specialists to run, what each is asked, and how they relate."""
    tasks: list[Task] = Field(
        description="One entry per specialist needed. For sequential mode, list the "
                    "one that must run first."
    )
    mode: Literal["parallel", "sequential"] = Field(
        description="Whether the second specialist depends on the first one's answer."
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_agents(cls, data):
        """Allow `Plan(agents=[...])` - a plan with no split, every task empty.

        The split is an addition, not a replacement: a caller that only knows
        which specialists to run still builds a valid plan, and every empty
        query degrades to the whole question at dispatch. Tests and any older
        caller construct plans this way.
        """
        if isinstance(data, dict) and "tasks" not in data and "agents" in data:
            data = dict(data)
            data["tasks"] = [{"agent": a, "query": ""} for a in data.pop("agents")]
        return data

    @property
    def agents(self) -> list[str]:
        return [t.agent for t in self.tasks]


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


# Two decisions in one prompt, in this order on purpose. Choosing the
# specialists comes first and stands alone; splitting the question is a second
# section that only applies once two of them have been chosen. Written the other
# way round - with the split up front and every example a decomposition - the
# model started adding a documentation specialist to plain counting questions to
# have something to split, and delegation accuracy fell from 90% to 50%.
PLAN_PROMPT = ChatPromptTemplate.from_template("""
You are the supervisor of an AWS AI/ML assistant. Two specialists are available.

- rag : AWS documentation. Explains how services work, their concepts, features,
        comparisons and best practices. It cannot count or rank anything.
- sql : A database of GitHub issues and Stack Overflow questions. Produces counts,
        rankings, trends and aggregates. It cannot explain how a service works.

Include a specialist only if the question cannot be answered without it. Adding
one that is not needed doubles the cost and dilutes the answer.

- A question that only asks for a count, a ranking or a trend is sql alone. Do
  not add rag to explain or give context to the number; the data specialist
  writes the sentence around its own result.
- A question that only asks what something is or how it works is rag alone. Do
  not add sql to say how often it is asked about.
- Two specialists only when the question has two parts, one of which no single
  specialist can answer.

mode (ignored when only one specialist is chosen):
- "parallel"   : both specialists can work now, because each half names what it
                 is about.
- "sequential" : the second specialist cannot form its query until the first has
                 answered, because the question identifies something by a
                 statistic and the identity is unknown until the data is queried.
                 List the specialist that resolves the unknown first - this is
                 almost always sql, since it is the one that ranks and counts.

Then write the query for each specialist you chose.

- One specialist: its query is the question, unchanged.
- Two: split the question. Each gets its own clause, written so it stands on its
  own, and never the clause meant for the other. A count that arrives carrying
  the documentation half of the question comes back filtered by it, which is a
  wrong number rather than a missing one.
- A clause asking what something is, does or is for belongs to rag however
  completely the rest of the question names its subject.
- In sequential mode, write the second query as best you can; it is rewritten
  with the concrete answer once the first specialist reports.

Examples:

  "What is Amazon Bedrock?"
    rag - documentation only, nothing to count.
      rag: "What is Amazon Bedrock?"

  "How many SageMaker questions were asked in 2023?"
    sql - a count, and the service is already named. Nothing here asks how
    SageMaker works, so nothing goes to rag.
      sql: "How many SageMaker questions were asked in 2023?"

  "Which AWS service has the most unanswered questions?"
    sql - a ranking. The answer is a name, not an explanation.
      sql: "Which AWS service has the most unanswered questions?"

  "What are the most common SageMaker issues, and how does training work?"
    sql + rag, mode "parallel" - two parts, and each half names its subject.
      sql: "What are the most commonly reported Amazon SageMaker issues?"
      rag: "How does training work in Amazon SageMaker?"

  "How many open issues does the aws-cdk repository have, and what is it for?"
    sql + rag, mode "parallel" - the repository being named does not make "what
    is it for" a database question.
      sql: "How many open issues does the aws-cdk repository have?"
      rag: "What is the aws-cdk repository for?"

  "Which service has the most questions and what does it do?"
    sql + rag, mode "sequential" - which service is unknown until the database is
    queried, and only then can the documentation be searched for it.
      sql: "Which AWS service has the most questions?"
      rag: "What does that service do?"

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

Change as little as possible. Replace the references and nothing else: do not
add a clause the user did not ask for, and do not enumerate the members of a set
the earlier turn named as a set. The user asked one question, and the rewrite
has to be that same question.

Then plan the dispatch for the rewritten question, not for the words the user
typed.

Two specialists are available.

- rag : AWS documentation. Explains how services work, their concepts, features,
        comparisons and best practices. It cannot count or rank anything.
- sql : A database of GitHub issues and Stack Overflow questions. Produces counts,
        rankings, trends and aggregates. It cannot explain how a service works.

Include a specialist only if the rewritten question cannot be answered without
it. Adding one that is not needed doubles the cost and dilutes the answer.

- A question that only asks for a count, a ranking or a trend is sql alone. Do
  not add rag to explain or give context to the number.
- A question that only asks what something is or how it works is rag alone. Do
  not add sql to say how often it is asked about.
- Two specialists only when the question has two parts, one of which no single
  specialist can answer.

mode (ignored when only one specialist is chosen):
- "parallel"   : both specialists can work now, because each half names what it
                 is about.
- "sequential" : the second specialist cannot form its query until the first has
                 answered. List the specialist that resolves the unknown first -
                 almost always sql, since it is the one that ranks and counts.

Then write the query for each specialist you chose, drawn from the rewritten
question.

- One specialist: its query is the rewritten question, unchanged.
- Two: split it. Each gets its own clause, written so it stands on its own, and
  never the clause meant for the other. A count that arrives carrying the
  documentation half comes back filtered by it - a wrong number, not a missing
  one.
- A clause asking what something is, does or is for belongs to rag however
  completely the rewrite names its subject. Resolving a reference into concrete
  names does not move that clause into the database.

Examples, given "What is Amazon Bedrock?" earlier in the conversation:

  "How much does it cost?"
    standalone "How much does Amazon Bedrock cost?"
      rag: "How much does Amazon Bedrock cost?"

  "How many questions are there about it?"
    standalone "How many questions are there about Amazon Bedrock?" - a count and
    nothing else, so rag is not involved.
      sql: "How many questions are there about Amazon Bedrock?"

  "What is Amazon Comprehend?"
    standalone "What is Amazon Comprehend?" - a new subject, nothing to resolve.
      rag: "What is Amazon Comprehend?"

  "How many questions are tagged with it, and how does it handle long documents?"
    standalone "How many questions are tagged amazon-bedrock, and how does Amazon
    Bedrock handle long documents?", mode "parallel"
      sql: "How many questions are tagged amazon-bedrock?"
      rag: "How does Amazon Bedrock handle long documents?"
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
        """Decide the dispatch, and what each specialist is asked.

        Falls back to rag-only if the model returns nothing usable.

        With earlier turns to draw on, the same call also resolves what the
        question refers to; `standalone_question` carries the result. Without
        them the call asks for the same schema Phase 1 asked for plus the split,
        so a single-specialist question takes the same path it always did.
        """
        try:
            if history:
                plan = self._followup_planner.invoke(FOLLOWUP_PROMPT.format_messages(
                    history=_render_history(history), question=question))
            else:
                plan = self._planner.invoke(PLAN_PROMPT.format_messages(question=question))
        except Exception:
            plan = None

        tasks: list[Task] = []
        if plan is not None:
            # Preserve order, drop duplicates and anything outside the vocabulary.
            seen = set()
            for t in plan.tasks:
                if t.agent in AGENTS and t.agent not in seen:
                    seen.add(t.agent)
                    tasks.append(Task(agent=t.agent, query=(t.query or "").strip()))

        # An empty or unusable rewrite degrades to the words the user typed,
        # which is exactly the pre-Phase-3 behaviour.
        standalone = (getattr(plan, "standalone_question", "") or "").strip() or question

        if not tasks:
            # RAG is the safer default: it says it lacks the information rather
            # than inventing a query against the database.
            return ContextualPlan(tasks=[Task(agent="rag", query=standalone)],
                                  mode="parallel", standalone_question=standalone)

        if len(tasks) == 1:
            # Nothing to split, so the split is not worth its risk: one
            # specialist answering the whole question is the case that has
            # always worked, and a paraphrase here could quietly drop a
            # constraint the user gave. The rewrite is only kept where it does
            # work - separating two specialists' halves.
            tasks = [Task(agent=tasks[0].agent, query=standalone)]

        return ContextualPlan(tasks=tasks, mode=plan.mode if plan else "parallel",
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
