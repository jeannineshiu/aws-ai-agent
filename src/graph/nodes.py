# src/graph/nodes.py
"""Graph nodes.

The specialist nodes stay thin adapters over the untouched pipelines in
src/rag/ and src/sql/. What changed in Phase 1 is that they answer the question
the supervisor hands them, which for the second half of a sequential plan is not
the question the user asked.
"""
from typing import Literal

from langgraph.types import Command

from src.graph.state import AgentState, Finding

MAX_PASSES = 4          # ceiling on supervisor dispatches; a loop needs a bound


def _effective_query(state: AgentState) -> str:
    """The question this specialist should answer.

    Normally the user's question. For the second specialist in a sequential plan
    it is the supervisor's rewrite, which carries the concrete entity the first
    specialist found.
    """
    return state.get("agent_query") or state["question"]


# ── supervisor ────────────────────────────────────────────────────────────────

def make_supervisor_node(supervisor, max_passes: int = MAX_PASSES):
    """Plan on the first pass, dispatch the remainder on later passes, then finish.

    Returns a Command so that updating state and choosing the next hop are one
    decision. `goto` takes a list, and a list of two is what fans out.
    """
    def supervisor_node(state: AgentState) -> Command[Literal["rag", "sql", "synthesize"]]:
        passes = state.get("passes", 0) + 1
        findings = state.get("findings", [])

        # Budget guard. Answer with whatever has been gathered rather than looping.
        if passes > max_passes:
            return Command(goto="synthesize", update={"passes": passes})

        # First pass: decide who runs and whether one depends on the other.
        if not findings:
            plan = supervisor.plan(state["question"])
            agents, mode = list(plan.agents), plan.mode

            if mode == "parallel" and len(agents) > 1:
                print(f"  -> Dispatch: {' + '.join(agents)} (parallel)")
                return Command(goto=agents, update={
                    "plan": [], "mode": "parallel",
                    "agent_query": state["question"], "passes": passes,
                })

            if len(agents) > 1:
                print(f"  -> Dispatch: {' -> '.join(agents)} (sequential)")
            else:
                print(f"  -> Dispatch: {agents[0]}")
            return Command(goto=[agents[0]], update={
                "plan": agents[1:], "mode": mode,
                "agent_query": state["question"], "passes": passes,
            })

        # Later passes: anything left in the plan runs now, with a query rewritten
        # from what has already come back.
        remaining = list(state.get("plan", []))
        if remaining:
            nxt = remaining[0]
            refined = supervisor.refine(state["question"], findings[-1], nxt)
            print(f"  -> Dispatch: {nxt} <- {refined[:60]!r}")
            return Command(goto=[nxt], update={
                "plan": remaining[1:], "agent_query": refined, "passes": passes,
            })

        return Command(goto="synthesize", update={"passes": passes})

    return supervisor_node


# ── specialists ───────────────────────────────────────────────────────────────

def make_rag_node(rag_pipeline):
    def rag_node(state: AgentState) -> dict:
        query = _effective_query(state)
        result = rag_pipeline.run(query)
        finding: Finding = {
            "agent": "rag",
            "query": query,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "retrieved_texts": result.get("retrieved_texts", []),
            "error": result.get("error"),
        }
        return {"findings": [finding]}
    return rag_node


def make_sql_node(sql_pipeline):
    def sql_node(state: AgentState) -> dict:
        query = _effective_query(state)
        result = sql_pipeline.run(query)
        finding: Finding = {
            "agent": "sql",
            "query": query,
            "answer": result.get("answer", ""),
            "sql": result.get("sql"),
            "data": result.get("data"),
            "error": result.get("error"),
        }
        return {"findings": [finding]}
    return sql_node


# ── synthesis ─────────────────────────────────────────────────────────────────

def make_synthesize_node(synthesizer):
    """Assemble the final result.

    `route` is derived from which specialists actually produced findings rather
    than predicted up front, so it reports what happened instead of what was
    intended.
    """
    def synthesize_node(state: AgentState) -> dict:
        by_agent = {f["agent"]: f for f in state.get("findings", [])}
        rag, sql = by_agent.get("rag"), by_agent.get("sql")

        if rag and sql:
            route = "both"
            answer = synthesizer.merge(state["question"], rag, sql)
        elif rag:
            route, answer = "rag", rag["answer"]
        elif sql:
            route, answer = "sql", sql["answer"]
        else:
            # No specialist ran. Unreachable through the wired edges; kept so the
            # node is total rather than raising on an empty findings list.
            route, answer = "rag", "Unable to process this question. Please try again."

        return {
            "route": route,
            "answer": answer,
            "citations": rag.get("citations", []) if rag else [],
            "data": sql.get("data") if sql else None,
            "sql": sql.get("sql") if sql else None,
        }
    return synthesize_node
