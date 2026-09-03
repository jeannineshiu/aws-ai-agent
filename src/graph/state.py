# src/graph/state.py
"""Shared state for the agent graph.

Phase 0 keeps this deliberately small: only the fields the ported v1 flow
actually reads or writes. Fields the later phases need (plan, critique,
attempt counters) are added when the node that uses them arrives, so the
schema never carries speculative keys.

The one forward-looking choice is `findings` being an accumulating channel
rather than a plain list. Two agents that run concurrently both write this
key, and without a reducer the second write silently replaces the first.
The `both` route is sequential in Phase 0, so the reducer is not load-bearing
yet — but making it a list-of-findings now is what lets Phase 1 fan out
without reshaping the state.
"""
import operator
from typing import Annotated, Any, TypedDict


class Finding(TypedDict, total=False):
    """One specialist's contribution to answering the question."""
    agent: str                      # "rag" | "sql"
    answer: str                     # natural-language answer from that pipeline
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
    route: str                                        # "rag" | "sql" | "both"
    findings: Annotated[list[Finding], operator.add]  # merges, never clobbers

    # composed output — mirrors the dict AWSAgent.run() returns
    answer: str
    citations: list[dict]
    data: Any
    sql: str | None
