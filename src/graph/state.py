# src/graph/state.py
"""Shared state for the agent graph.

Fields are added when the node that uses them arrives, so the schema never
carries speculative keys. Phase 1 added the supervisor's dispatch bookkeeping;
Phase 2 adds one counter per quality loop.

Every loop counter exists for the same reason: a cycle without a ceiling is a
hang. The supervisor resets the per-specialist budgets each time it dispatches,
so a re-dispatch gets a fresh allowance while `passes` and `revisions` bound the
outer loops.

`findings` is an accumulating channel rather than a plain list. Under parallel
dispatch both specialists write this key in the same superstep, and without the
reducer the second write silently replaces the first. Phase 0 was sequential so
it never exercised that; Phase 1 does.

Phase 3 adds the second turn. With a checkpointer the state survives the turn
that produced it, which is the point for `history` and a bug for everything
else: `findings` would keep the last turn's evidence and `passes` would start
the new turn already over budget. So the per-turn channels are reset by the
caller (`GraphAgent._turn_input`), and `findings` gets a reducer that
understands a reset — a plain `operator.add` channel has no way to express one.
"""
import operator
from typing import Annotated, Any, TypedDict


def merge_findings(left: list | None, right: list | None) -> list:
    """Accumulate within a turn; `None` clears the channel for the next one.

    Every write from a node is a list and appends, so parallel dispatch still
    merges. `None` is reserved for the turn boundary, and only the caller
    writes it.
    """
    if right is None:
        return []
    return (left or []) + list(right)


class Finding(TypedDict, total=False):
    """One specialist's contribution to answering the question."""
    agent: str                      # "rag" | "sql"
    answer: str                     # natural-language answer from that pipeline
    query: str                      # the question this specialist actually answered
    citations: list[dict]           # RAG only
    retrieved_texts: list[str]      # RAG only — kept for evaluation
    sql: str | None

    # human-in-the-loop
    pending_sql: str | None      # query held at the confirmation gate
    confirm_reason: str          # why it is being held
    sql_declined: bool           # the human refused it; do not retry, report it                 # SQL only — the generated query
    data: Any                       # SQL only — pandas DataFrame or None
    error: str | None               # set when the pipeline reported a failure


class AgentState(TypedDict, total=False):
    """Channels the graph reads and writes.

    `question` is the only required input; everything else is produced by a node.
    """
    question: str
    findings: Annotated[list[Finding], merge_findings]  # merges, never clobbers

    # multi-turn
    history: Annotated[list[dict], operator.add]  # {question, answer} per finished turn
    resolved_question: str   # the question with references to earlier turns filled in

    # supervisor bookkeeping
    plan: list[str]          # specialists still to dispatch (sequential only)
    mode: str                # "parallel" | "sequential"
    agent_queries: dict[str, str]  # per specialist, the question it alone should answer
                             # (the supervisor is the only writer, so a plain
                             #  channel is enough even under parallel fan-out)
    awaiting: list[str]      # specialists dispatched and not yet reported
    passes: int              # dispatch budget guard — a supervisor loop needs a ceiling

    # corrective retrieval loop
    rag_attempts: int
    rag_query: str | None    # rewritten search query; None means "not retrying"
    rag_missing: str         # what the grader said the documents lacked

    # SQL repair loop
    sql_attempts: int
    sql_error: str | None    # exception text, or the note that a query returned nothing
    last_sql: str | None     # the query that failed, fed back to the repairer

    # critic loop
    revisions: int
    critique: str | None     # claims the critic found unsupported
    grounded: bool | None    # the critic's verdict, or None when it never ran

    # composed output — mirrors the dict AWSAgent.run() returns
    route: str               # "rag" | "sql" | "both", derived from who actually ran
    answer: str
    citations: list[dict]
    data: Any
    sql: str | None

    # human-in-the-loop
    pending_sql: str | None      # query held at the confirmation gate
    confirm_reason: str          # why it is being held
    sql_declined: bool           # the human refused it; do not retry, report it
