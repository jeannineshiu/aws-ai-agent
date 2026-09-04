# src/graph/nodes.py
"""Graph nodes.

Phase 2 gives two of these nodes a loop of their own. Both loops exist because
of a failure the evaluation harness actually produced, not because a graph can
express cycles:

  - retrieval returned four chunks from a single page for a question that
    deserved several sources, and nothing downstream noticed;
  - a generated query used `<bedrock>` for a tag written `<amazon-bedrock>`,
    returned zero rows, and the zero was reported as the answer.

Every loop has a ceiling. The supervisor resets the per-specialist budgets on
each dispatch, so a re-dispatch gets a fresh allowance while `passes` bounds the
outer one.
"""
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, interrupt

from src.graph.repair import looks_empty
from src.graph.state import AgentState, Finding

MAX_PASSES = 4              # supervisor dispatches per turn
MAX_RAG_ATTEMPTS = 2        # retrievals per dispatch, including the first
MAX_SQL_ATTEMPTS = 2        # query attempts per dispatch, including the first
MAX_REVISIONS = 1           # critic rejections honoured per turn


def _question(state: AgentState) -> str:
    """The turn's question, with references to earlier turns already resolved.

    Every node downstream of the supervisor reads this instead of `question`.
    On a first turn the two are the same string; on a follow-up, `question` is
    still "how much does it cost" and nothing but the supervisor can act on
    that. `question` is kept unchanged so the transcript records what the user
    actually typed.
    """
    return state.get("resolved_question") or state["question"]


def _effective_query(state: AgentState, agent: str) -> str:
    """The question this specialist should answer.

    The supervisor splits a two-part question so each specialist gets its own
    half; this is where the half is picked up. For the second specialist in a
    sequential plan it is the supervisor's rewrite, which carries the concrete
    entity the first specialist found.

    Falls back to the whole question, which is what every specialist got before
    the plan carried a split, and what a plan whose split came back empty still
    gets.
    """
    return (state.get("agent_queries") or {}).get(agent) or _question(state)


def _fresh_budgets() -> dict:
    """Per-specialist loop counters, reset on every dispatch."""
    return {
        "rag_attempts": 0, "rag_query": None, "rag_missing": "",
        "sql_attempts": 0, "sql_error": None, "last_sql": None,
    }


# ── supervisor ────────────────────────────────────────────────────────────────

def make_supervisor_node(supervisor, max_passes: int = MAX_PASSES):
    """Plan on the first pass, dispatch the remainder on later passes, then finish.

    The `awaiting` guard is not bookkeeping for its own sake. Each specialist
    edges back here, so under parallel dispatch this node is woken once per
    specialist — and once a specialist can loop, the two branches no longer
    finish in the same superstep. Without the guard the supervisor decides the
    turn is over while the slower branch is still retrieving: its finding is
    dropped, and the answer is built from half the evidence.
    """
    def supervisor_node(state: AgentState) -> Command[Literal["rag", "sql", "synthesize"]]:
        findings = state.get("findings", [])
        awaiting = state.get("awaiting") or []

        # Woken by one branch while another is still working. Park this wake-up;
        # the last specialist to finish wakes us again with everything.
        if awaiting and not set(awaiting) <= {f["agent"] for f in findings}:
            return Command(goto=[])

        passes = state.get("passes", 0) + 1

        # Budget guard. Answer with whatever has been gathered rather than looping.
        if passes > max_passes:
            return Command(goto="synthesize", update={"passes": passes, "awaiting": []})

        if not findings:
            # History is only non-empty when a checkpointer carried the thread
            # forward, so a single-turn caller takes exactly the Phase 1 path.
            plan = supervisor.plan(state["question"], state.get("history"))
            agents, mode = list(plan.agents), plan.mode
            question = getattr(plan, "standalone_question", "") or state["question"]
            if question != state["question"]:
                print(f"  -> Resolved: {question[:72]!r}")

            # The split, where the plan carries one. Empty entries are left out
            # rather than filled in here, so `_effective_query` decides what a
            # missing half degrades to, in one place.
            queries = {t.agent: t.query.strip() for t in getattr(plan, "tasks", [])
                       if (t.query or "").strip()}
            for agent in agents:
                if queries.get(agent) and queries[agent] != question:
                    print(f"       {agent} <- {queries[agent][:60]!r}")

            if mode == "parallel" and len(agents) > 1:
                print(f"  -> Dispatch: {' + '.join(agents)} (parallel)")
                return Command(goto=agents, update={
                    "plan": [], "mode": "parallel", "awaiting": agents,
                    "resolved_question": question,
                    "agent_queries": queries, "passes": passes,
                    **_fresh_budgets(),
                })

            if len(agents) > 1:
                print(f"  -> Dispatch: {' -> '.join(agents)} (sequential)")
            else:
                print(f"  -> Dispatch: {agents[0]}")
            return Command(goto=[agents[0]], update={
                "plan": agents[1:], "mode": mode, "awaiting": [agents[0]],
                "resolved_question": question,
                "agent_queries": queries, "passes": passes,
                **_fresh_budgets(),
            })

        remaining = list(state.get("plan", []))
        if remaining:
            nxt = remaining[0]
            queries = dict(state.get("agent_queries") or {})
            # Refine the half this specialist was given, not the whole question:
            # its half is what is left to answer, and the other half has already
            # been answered by the finding being fed in.
            refined = supervisor.refine(queries.get(nxt) or _question(state),
                                        findings[-1], nxt)
            print(f"  -> Dispatch: {nxt} <- {refined[:60]!r}")
            return Command(goto=[nxt], update={
                "plan": remaining[1:], "agent_queries": {**queries, nxt: refined},
                "passes": passes, "awaiting": [nxt], **_fresh_budgets(),
            })

        return Command(goto="synthesize", update={"passes": passes, "awaiting": []})

    return supervisor_node


# ── RAG specialist, with corrective retrieval ─────────────────────────────────

def make_rag_node(rag_pipeline, grader=None, max_attempts: int = MAX_RAG_ATTEMPTS):
    """Retrieve, grade what came back, and search again with better wording if it
    will not do.

    The rewritten query is used for retrieval only. Generation always answers the
    original question — the rewrite is a search term, not a change of subject.
    """
    def rag_node(state: AgentState) -> dict:
        question = _effective_query(state, "rag")
        attempts = state.get("rag_attempts", 0)
        search_query = state.get("rag_query") or question

        docs = rag_pipeline.retrieve(search_query)

        if grader is not None and attempts + 1 < max_attempts:
            grade = grader.grade(question, docs)
            if not grade.sufficient:
                rewritten = grader.rewrite(question, search_query, grade.missing)
                print(f"  -> Retrieval insufficient ({grade.missing[:48]!r}); "
                      f"retrying as {rewritten[:48]!r}")
                return {
                    "rag_attempts": attempts + 1,
                    "rag_query": rewritten,
                    "rag_missing": grade.missing,
                }

        result = rag_pipeline.generate(question, docs)
        finding: Finding = {
            "agent": "rag",
            "query": question,
            "search_query": search_query,
            "attempts": attempts + 1,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "retrieved_texts": result.get("retrieved_texts", []),
            "error": result.get("error"),
        }
        return {"findings": [finding], "rag_attempts": attempts + 1, "rag_query": None}

    return rag_node


def after_rag(state: AgentState) -> Literal["rag", "supervisor"]:
    """A pending rewrite means the node asked for another retrieval."""
    return "rag" if state.get("rag_query") else "supervisor"


# ── SQL specialist, with repair ───────────────────────────────────────────────

def _permitted(sql_pipeline, sql: str, approved: bool) -> tuple[bool, str]:
    """May this query reach the database?

    `approved` means a human has just said yes, which lifts the reviewer's
    middle tier and nothing else. An approval is an answer to a question the
    code could not decide; a rejection was never a question.
    """
    review = getattr(sql_pipeline, "review_sql", None)
    if review is None:
        return sql_pipeline.validate_sql(sql)     # a pipeline without the tiers

    verdict = review(sql)
    if verdict.verdict == "allow":
        return True, "OK"
    if verdict.verdict == "confirm" and approved:
        return True, verdict.reason
    return False, verdict.reason


def _confirmation_reason(sql_pipeline, sql: str, attempts: int) -> str | None:
    """Why a human should see this query before it runs, or None.

    Two cases, and neither is "all SQL is dangerous" - the database is a
    read-only local file and a blanket confirmation on every query would be
    friction that teaches people to click through it.

    The first is the reviewer's middle tier: the query is not clearly a read and
    not clearly a write, and code that cannot classify it should not decide it.

    The second is a repair. The repairer rewrote the query after the first one
    failed, and its prompt has to be told not to widen a filter just to turn a
    zero into a number. That is exactly the mistake a person spots in one look
    at the SQL, and until now it ran without anyone seeing it.
    """
    review = getattr(sql_pipeline, "review_sql", None)
    if review is not None:
        verdict = review(sql)
        if verdict.verdict == "reject":
            # Not a question for a human. Fail it now and let the repair loop
            # try again rather than asking someone to approve a dead query.
            return None
        if verdict.verdict == "confirm":
            return verdict.reason
    if attempts > 0:
        return "it is a rewrite of a query that came back empty, not the one your question produced"
    return None


def make_sql_node(sql_pipeline, repairer=None, max_attempts: int = MAX_SQL_ATTEMPTS,
                  confirm: bool = False):
    """Generate, validate, execute — and on failure feed the failure back.

    Finding nothing counts as a failure — including a COUNT that comes back as
    a single row containing zero, which is the case this loop was built for and
    which a plain "no rows" check misses entirely.

    With `confirm` on, a query the reviewer cannot classify - or one the
    repairer wrote rather than the user's question - stops here for a human.
    The pause takes a second pass through this node on purpose: `interrupt()`
    replays its node from the top when the answer arrives, so it has to be the
    first thing that happens, or resuming would re-run the LLM call that
    produced the query and could approve one query while executing another.
    """
    def sql_node(state: AgentState) -> dict:
        question = _effective_query(state, "sql")
        attempts = state.get("sql_attempts", 0)
        pending = state.get("pending_sql")

        if pending:
            decision = interrupt({
                "sql": pending,
                "question": question,
                "reason": state.get("confirm_reason", ""),
            })
            approved = decision.get("approved") if isinstance(decision, dict) else bool(decision)
            if not approved:
                finding: Finding = {
                    "agent": "sql", "query": question, "attempts": attempts + 1,
                    "answer": "The query needed to answer this was not run.",
                    "sql": pending, "data": None,
                    "error": "declined before execution",
                }
                return {"findings": [finding], "pending_sql": None,
                        "confirm_reason": "", "sql_declined": True,
                        "sql_error": None, "sql_attempts": attempts + 1}
            sql = pending
        else:
            prev_error, prev_sql = state.get("sql_error"), state.get("last_sql")

            if prev_error and prev_sql and repairer is not None:
                sql = repairer.repair(question, prev_sql, prev_error,
                                      conn=getattr(sql_pipeline, "conn", None))
            else:
                sql = sql_pipeline.generate_sql(question)

            if confirm:
                reason = _confirmation_reason(sql_pipeline, sql, attempts)
                if reason:
                    # Nothing has touched the database yet. Park the query and
                    # come back into this node to ask.
                    return {"pending_sql": sql, "confirm_reason": reason}

        df, error = None, None
        ok, reason = _permitted(sql_pipeline, sql, approved=bool(pending))
        if not ok:
            error = f"rejected by the validator: {reason}"
        else:
            try:
                df = sql_pipeline.execute_sql(sql)
                if looks_empty(df):
                    # Keep the result: if the repair does no better, this is
                    # still the answer, and zero is sometimes correct.
                    error = "the query ran but found nothing"
            except Exception as e:
                error = str(e)

        # Retrying without a repairer would re-run generate_sql on the same
        # question and produce the same query — a wasted attempt, not a repair.
        if error and repairer is not None and attempts + 1 < max_attempts:
            print(f"  -> SQL failed ({error[:56]}); repairing")
            return {"sql_attempts": attempts + 1, "sql_error": error,
                    "last_sql": sql, "pending_sql": None, "confirm_reason": ""}

        if df is not None:
            answer = sql_pipeline.explain_results(question, df)
        else:
            answer = f"Could not answer from the data: {error}"

        finding: Finding = {
            "agent": "sql",
            "query": question,
            "attempts": attempts + 1,
            "answer": answer,
            "sql": sql,
            "data": df,
            "error": error,
        }
        return {"findings": [finding], "sql_attempts": attempts + 1,
                "sql_error": None, "pending_sql": None, "confirm_reason": ""}

    return sql_node


def after_sql(state: AgentState) -> Literal["sql", "supervisor"]:
    """A retained error means the node asked for a repair attempt; a parked
    query means it asked to come back and put the question to a human."""
    return "sql" if (state.get("sql_error") or state.get("pending_sql")) else "supervisor"


# ── synthesis ─────────────────────────────────────────────────────────────────

def _contexts(state: AgentState) -> list[str]:
    """Everything the answer could legitimately be grounded in.

    Retrieved documents are only half of it. On a `both` route the factual core
    of the answer often comes from the query result, and judging that against
    the documentation alone rejects it for not appearing somewhere it was never
    going to appear. The first version of this did exactly that: the critic
    threw away a correct "aws-neuron/aws-neuron-sdk has 79 open issues" because
    no AWS document mentions it, and the redraft replaced it with "I cannot
    answer". Faithfulness rose because the answer no longer claimed anything;
    answer relevancy collapsed for the same reason.
    """
    out = []
    for f in state.get("findings", []):
        out.extend(f.get("retrieved_texts") or [])

        if f.get("agent") == "sql" and f.get("data") is not None:
            data = f["data"]
            try:
                rendered = data.to_string(index=False)
            except AttributeError:
                rendered = str(data)
            out.append(f"Query: {f.get('sql') or ''}\nResult:\n{rendered[:1500]}")
    return out


def make_synthesize_node(synthesizer):
    """Assemble the final result, or redraft it when the critic pushed back.

    `route` is derived from which specialists actually produced findings rather
    than predicted up front, so it reports what happened instead of what was
    intended.
    """
    def synthesize_node(state: AgentState) -> dict:
        # A later finding from the same specialist supersedes an earlier one.
        by_agent = {f["agent"]: f for f in state.get("findings", [])}
        rag, sql = by_agent.get("rag"), by_agent.get("sql")
        critique = state.get("critique")

        if rag and sql:
            route = "both"
            answer = synthesizer.merge(_question(state), rag, sql)
        elif rag:
            route, answer = "rag", rag["answer"]
        elif sql:
            route, answer = "sql", sql["answer"]
        else:
            # No specialist ran. Unreachable through the wired edges; kept so the
            # node is total rather than raising on an empty findings list.
            route, answer = "rag", "Unable to process this question. Please try again."

        if critique:
            print(f"  -> Redrafting: {critique[:64]!r}")
            answer = synthesizer.revise(_question(state), answer,
                                        _contexts(state), critique)

        return {
            "route": route,
            "answer": answer,
            "citations": rag.get("citations", []) if rag else [],
            "data": sql.get("data") if sql else None,
            "sql": sql.get("sql") if sql else None,
            "critique": None,
        }
    return synthesize_node


# ── critic ────────────────────────────────────────────────────────────────────

def make_critic_node(critic=None, max_revisions: int = MAX_REVISIONS):
    """Check the answer against the evidence before it leaves.

    RAGAS Faithfulness, moved from the offline harness into the request path.
    Skipped when there is no retrieved evidence to check against: a SQL-only
    answer is grounded in a DataFrame the pipeline computed, so there is no
    hallucination surface worth an LLM call.
    """
    def critic_node(state: AgentState) -> Command[Literal["synthesize", "remember"]]:
        revisions = state.get("revisions", 0)
        contexts = _contexts(state)

        # No update on the skip path, a verdict on the others. The difference is
        # what lets a caller tell "the answer was checked and passed" from "no
        # check happened" - which, before Phase 3 put the trajectory in front of
        # a user, nothing had ever needed to distinguish.
        if critic is None or not contexts or revisions >= max_revisions:
            return Command(goto="remember")

        verdict = critic.check(_question(state), state.get("answer", ""), contexts)
        if verdict.grounded:
            return Command(goto="remember", update={"grounded": True})

        print(f"  -> Critic rejected: {verdict.unsupported[:64]!r}")
        return Command(goto="synthesize", update={
            "revisions": revisions + 1,
            "grounded": False,
            "critique": verdict.unsupported or "claims not supported by the sources",
        })

    return critic_node


# ── remember ──────────────────────────────────────────────────────────────────

def make_remember_node():
    """Record the finished turn, so the next one can refer back to it.

    A separate node rather than a write from `synthesize` because synthesis can
    run twice in a turn: the critic sends a rejected answer back to be
    redrafted, and a history write there would file both drafts as if the user
    had asked twice. This node sits on the graph's single exit, so it runs once
    per turn by construction.

    The transcript stores what the user typed, not the resolved rewrite. The
    supervisor resolves the next follow-up against the conversation as it
    happened; feeding it back its own rewrites would let one bad resolution
    fix itself into the record.
    """
    def remember_node(state: AgentState) -> dict:
        return {"history": [{
            "question": state["question"],
            "answer": state.get("answer", ""),
        }]}

    return remember_node
