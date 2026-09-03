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
"""
import operator
from typing import Annotated, Any, TypedDict


class Finding(TypedDict, total=False):
    """One specialist's contribution to answering the question."""
    agent: str                      # "rag" | "sql"
    answer: str                     # natural-language answer from that pipeline
    query: str                      # the question this specialist actually answered
    citations: list[dict]           # RAG only
    retrieved_texts: list[str]      # RAG only — kept for evaluation
    sql: str | None                 # SQL only — the generated query
    data: Any                       # SQL only — pandas DataFrame or None
    error: str | None               # set when the pipeline reported a failure


class AgentState(TypedDict, total=False):
    """Channels the graph reads and writes.

    `question` is the only required input; everything else is produced by a node.
    """
    question: str
    findings: Annotated[list[Finding], operator.add]  # merges, never clobbers

    # supervisor bookkeeping
    plan: list[str]          # specialists still to dispatch (sequential only)
    mode: str                # "parallel" | "sequential"
    agent_query: str         # the question the next dispatched specialist should answer
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

    # composed output — mirrors the dict AWSAgent.run() returns
    route: str               # "rag" | "sql" | "both", derived from who actually ran
    answer: str
    citations: list[dict]
    data: Any
    sql: str | None
