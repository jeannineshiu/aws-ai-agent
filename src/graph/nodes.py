# src/graph/nodes.py
"""Graph nodes for the Phase 0 port.

Each node is a thin adapter: it calls the existing pipeline unchanged and
writes the result into state as a Finding. No pipeline logic lives here, and
none of it was modified — src/rag/pipeline.py and src/sql/pipeline.py are
byte-identical to the v1 versions.
"""
from src.graph.state import AgentState, Finding
from src.router.router import RouteType


def make_route_node(router):
    """Classify the question. Mirrors AWSAgent.run() step 1."""
    def route_node(state: AgentState) -> dict:
        route = router.route(state["question"])
        value = route.value if isinstance(route, RouteType) else str(route)
        print(f"  -> Route: {value}")
        return {"route": value}
    return route_node


def make_rag_node(rag_pipeline):
    def rag_node(state: AgentState) -> dict:
        if state.get("route") == "both":
            print("  -> Running RAG...")
        result = rag_pipeline.run(state["question"])
        finding: Finding = {
            "agent": "rag",
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "retrieved_texts": result.get("retrieved_texts", []),
            "error": result.get("error"),
        }
        return {"findings": [finding]}
    return rag_node


def make_sql_node(sql_pipeline):
    def sql_node(state: AgentState) -> dict:
        if state.get("route") == "both":
            print("  -> Running SQL...")
        result = sql_pipeline.run(state["question"])
        finding: Finding = {
            "agent": "sql",
            "answer": result.get("answer", ""),
            "sql": result.get("sql"),
            "data": result.get("data"),
            "error": result.get("error"),
        }
        return {"findings": [finding]}
    return sql_node


def compose_node(state: AgentState) -> dict:
    """Assemble the final result from whatever findings were produced.

    Phase 0 reproduces AWSAgent.run() exactly, including the f-string join for
    the `both` route. That join is a known defect — the two answers are stapled
    together rather than reconciled — but fixing it here would make any change
    in the evaluation numbers ambiguous between "the port" and "the fix".
    It is Phase 1's job.
    """
    by_agent = {f["agent"]: f for f in state.get("findings", [])}
    rag = by_agent.get("rag")
    sql = by_agent.get("sql")

    if rag and sql:
        answer = (
            f"**From documentation:**\n{rag['answer']}\n\n"
            f"**From data analysis:**\n{sql['answer']}"
        )
    elif rag:
        answer = rag["answer"]
    elif sql:
        answer = sql["answer"]
    else:
        # No specialist ran. Unreachable through the wired edges; kept so the
        # node is total rather than raising on an empty findings list.
        answer = "Unable to process this question. Please try again."

    return {
        "answer": answer,
        "citations": rag.get("citations", []) if rag else [],
        "data": sql.get("data") if sql else None,
        "sql": sql.get("sql") if sql else None,
    }


# ── conditional edges ────────────────────────────────────────────────────────

def after_route(state: AgentState) -> str:
    """rag and both both start at the RAG node; both continues to SQL afterwards.

    An unrecognised route falls through to "rag", matching QueryRouter's own
    fallback: RAG will say it lacks the information rather than inventing SQL.
    """
    return "sql" if state.get("route") == "sql" else "rag"


def after_rag(state: AgentState) -> str:
    """Only the `both` route continues into SQL, preserving v1's RAG-then-SQL order."""
    return "sql" if state.get("route") == "both" else "compose"
