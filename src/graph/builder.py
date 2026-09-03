# src/graph/builder.py
"""Wire the supervisor graph.

                  +-------------------------------+
                  |                               | findings
                  v                               |
    START --> supervisor --+--> rag --------------+
                  |        |                      |
                  |        +--> sql --------------+
                  |
                  +--> synthesize --> END

The supervisor dispatches with Command(goto=...). A list of two specialists
fans out and both run in the same superstep; a list of one runs alone and the
supervisor sees its finding before choosing what follows.

The edges back into the supervisor are the dispatch loop - the mechanism that
lets a plan be decided in stages rather than all at once. The quality loops
(regrade and rewrite inside a specialist, SQL repair, a critic that can reject
an answer) are Phase 2 and are not here yet.
"""
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    MAX_PASSES,
    make_rag_node,
    make_sql_node,
    make_supervisor_node,
    make_synthesize_node,
)
from src.graph.state import AgentState


def build_graph(supervisor, rag_pipeline, sql_pipeline, synthesizer,
                checkpointer=None, max_passes: int = MAX_PASSES):
    """Compile the graph from injected collaborators.

    Everything is a parameter so tests can pass fakes and exercise dispatch,
    fan-out and synthesis without any API calls.
    """
    g = StateGraph(AgentState)

    g.add_node("supervisor", make_supervisor_node(supervisor, max_passes))
    g.add_node("rag", make_rag_node(rag_pipeline))
    g.add_node("sql", make_sql_node(sql_pipeline))
    g.add_node("synthesize", make_synthesize_node(synthesizer))

    g.add_edge(START, "supervisor")
    # supervisor -> {rag, sql, synthesize} is declared by the Command return type
    g.add_edge("rag", "supervisor")
    g.add_edge("sql", "supervisor")
    g.add_edge("synthesize", END)

    return g.compile(checkpointer=checkpointer)


class GraphAgent:
    """Drop-in replacement for AWSAgent, backed by the supervisor graph.

    run() returns the same dict shape v1 returns, so app.py and any evaluation
    harness can swap implementations by changing the constructor.
    """

    def __init__(self, supervisor=None, rag_pipeline=None, sql_pipeline=None,
                 synthesizer=None, checkpointer=None, max_passes: int = MAX_PASSES):
        # Imported lazily so tests can build a graph from fakes without touching
        # Chroma, SQLite or the OpenAI client.
        if None in (supervisor, rag_pipeline, sql_pipeline, synthesizer):
            from src.graph.supervisor import Supervisor
            from src.graph.synthesizer import Synthesizer
            from src.rag.pipeline import RAGPipeline
            from src.sql.pipeline import SQLPipeline

            print("Initializing AWS AI/ML Agent (graph)...")
            supervisor = supervisor or Supervisor()
            rag_pipeline = rag_pipeline or RAGPipeline()
            sql_pipeline = sql_pipeline or SQLPipeline()
            synthesizer = synthesizer or Synthesizer()
            print("Agent ready.")

        self.graph = build_graph(supervisor, rag_pipeline, sql_pipeline,
                                 synthesizer, checkpointer, max_passes)

    def run_state(self, question: str, config: dict | None = None) -> dict:
        """Full final state, including every finding and the query it answered.

        run() projects this down to v1's dict shape. Callers that need to see
        what the graph actually did - tracing, evaluation, the parity script -
        use this instead.
        """
        return self.graph.invoke({"question": question}, config or {})

    def run(self, question: str, config: dict | None = None) -> dict:
        final = self.run_state(question, config)
        return {
            "question": question,
            "route": final.get("route", "rag"),
            "answer": final.get("answer", ""),
            "citations": final.get("citations", []),
            "data": final.get("data"),
            "sql": final.get("sql"),
        }
