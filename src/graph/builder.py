# src/graph/builder.py
"""Wire the Phase 0 graph.

Topology is identical in behaviour to AWSAgent.run()'s if/elif:

    START -> route -+-- "sql" -----------------> sql -----> compose -> END
                    |
                    +-- "rag"  --> rag ------------------> compose -> END
                    |
                    +-- "both" --> rag --> sql ---------> compose -> END

Every edge points forward. That is the point of Phase 0: the machinery is a
graph, the behaviour is not yet. Cycles arrive in Phase 2, once there are
measured failures to justify them.
"""
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    after_rag,
    after_route,
    compose_node,
    make_rag_node,
    make_route_node,
    make_sql_node,
)
from src.graph.state import AgentState


def build_graph(router, rag_pipeline, sql_pipeline, checkpointer=None):
    """Compile the graph from injected collaborators.

    The pipelines are parameters rather than constructed here so tests can pass
    fakes and exercise the topology without any API calls.
    """
    g = StateGraph(AgentState)

    g.add_node("route", make_route_node(router))
    g.add_node("rag", make_rag_node(rag_pipeline))
    g.add_node("sql", make_sql_node(sql_pipeline))
    g.add_node("compose", compose_node)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, {"rag": "rag", "sql": "sql"})
    g.add_conditional_edges("rag", after_rag, {"sql": "sql", "compose": "compose"})
    g.add_edge("sql", "compose")
    g.add_edge("compose", END)

    return g.compile(checkpointer=checkpointer)


class GraphAgent:
    """Drop-in replacement for AWSAgent, backed by the graph.

    run() returns the same dict shape and the same values, so app.py and any
    evaluation harness can swap between v1 and v2 by changing the constructor.
    """

    def __init__(self, router=None, rag_pipeline=None, sql_pipeline=None, checkpointer=None):
        # Imported lazily so tests can build a graph from fakes without
        # touching Chroma, SQLite or the OpenAI client.
        if router is None or rag_pipeline is None or sql_pipeline is None:
            from src.rag.pipeline import RAGPipeline
            from src.router.router import QueryRouter
            from src.sql.pipeline import SQLPipeline

            print("Initializing AWS AI/ML Agent (graph)...")
            router = router or QueryRouter()
            rag_pipeline = rag_pipeline or RAGPipeline()
            sql_pipeline = sql_pipeline or SQLPipeline()
            print("Agent ready.")

        self.graph = build_graph(router, rag_pipeline, sql_pipeline, checkpointer)

    def run(self, question: str, config: dict | None = None) -> dict:
        final = self.graph.invoke({"question": question}, config or {})
        return {
            "question": question,
            "route": final.get("route", "rag"),
            "answer": final.get("answer", ""),
            "citations": final.get("citations", []),
            "data": final.get("data"),
            "sql": final.get("sql"),
        }
