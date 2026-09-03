# src/graph/builder.py
"""Wire the supervisor graph.

                       +--------------------------------+
                       |                                | findings
              regrade  v                                |
                +--> rag ----------------------------->-+
                |    ^  |                               |
                |    +--+ insufficient, rewrite query   |
    START --> supervisor                                |
                |    +--+ failed or empty, repair       |
                |    v  |                               |
                +--> sql ----------------------------->-+
                |
                +--> synthesize --> critic --> remember --> END
                         ^             |
                         +-------------+ not grounded, redraft

Three loops, each with a ceiling, each built for a failure the evaluation
harness produced:

  rag -> rag         retrieval that cannot answer the question is searched
                     again with wording drawn from AWS documentation
  sql -> sql         a query that errors, is rejected by the validator, or
                     returns nothing is rewritten with the failure as evidence
  critic -> synth    an answer that outruns its sources is redrafted against them

The supervisor loop from Phase 1 is a dispatch loop: it decides what to do next.
These are quality loops: they decide whether what just happened was good enough.

`remember` is the graph's single exit. It files the finished turn in `history`,
which is the only channel a checkpointer is meant to carry across turns - see
`GraphAgent._turn_input` for the ones that must not.
"""
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    MAX_PASSES,
    MAX_RAG_ATTEMPTS,
    MAX_REVISIONS,
    MAX_SQL_ATTEMPTS,
    after_rag,
    after_sql,
    make_critic_node,
    make_rag_node,
    make_remember_node,
    make_sql_node,
    make_supervisor_node,
    make_synthesize_node,
)
from src.graph.state import AgentState


def build_graph(supervisor, rag_pipeline, sql_pipeline, synthesizer,
                grader=None, repairer=None, critic=None, checkpointer=None,
                max_passes: int = MAX_PASSES,
                max_rag_attempts: int = MAX_RAG_ATTEMPTS,
                max_sql_attempts: int = MAX_SQL_ATTEMPTS,
                max_revisions: int = MAX_REVISIONS):
    """Compile the graph from injected collaborators.

    Everything is a parameter so tests can pass fakes and exercise dispatch,
    fan-out, the loops and synthesis without any API calls. Passing None for
    grader, repairer or critic disables that loop, which is how the Phase 1
    behaviour stays reachable for comparison.
    """
    g = StateGraph(AgentState)

    g.add_node("supervisor", make_supervisor_node(supervisor, max_passes))
    g.add_node("rag", make_rag_node(rag_pipeline, grader, max_rag_attempts))
    g.add_node("sql", make_sql_node(sql_pipeline, repairer, max_sql_attempts))
    g.add_node("synthesize", make_synthesize_node(synthesizer))
    g.add_node("critic", make_critic_node(critic, max_revisions))
    g.add_node("remember", make_remember_node())

    g.add_edge(START, "supervisor")
    # supervisor -> {rag, sql, synthesize} is declared by the Command return type
    g.add_conditional_edges("rag", after_rag, {"rag": "rag", "supervisor": "supervisor"})
    g.add_conditional_edges("sql", after_sql, {"sql": "sql", "supervisor": "supervisor"})
    g.add_edge("synthesize", "critic")
    # critic -> {synthesize, remember} is declared by the Command return type
    g.add_edge("remember", END)

    return g.compile(checkpointer=checkpointer)


class GraphAgent:
    """Drop-in replacement for AWSAgent, backed by the supervisor graph.

    run() returns the same dict shape v1 returns, so app.py and any evaluation
    harness can swap implementations by changing the constructor.
    """

    #   loops="repair"  SQL repair only. The default, because it is the only
    #                    configuration the evaluation supports paying for.
    #   loops="all"      every quality loop, including grading and the critic.
    #   loops=False      the Phase 1 graph, for comparison.
    #
    # Measured over 30 ground-truth samples (scripts/run_evaluation.py):
    #
    #                        v1     no loops   repair only   all loops
    #   answer_relevancy   0.487      0.699        0.724        0.610
    #   faithfulness       0.820      0.808        0.848        0.854
    #   adversarial SQL      60%        60%         100%         100%
    #   LLM calls/query     2.87       3.43         3.73         5.93
    #
    # The grader and the critic together add 2.2 calls per query and move
    # faithfulness by 0.006 - inside the run-to-run noise band - while costing
    # 0.114 of answer relevancy, because the critic trims content it cannot tie
    # to a source. They stay in the tree behind the flag: a differently shaped
    # question set could justify them, this one does not.
    #
    # `memory=True` compiles with a checkpointer so a thread can be continued.
    # It is off by default because a compiled checkpointer makes `thread_id`
    # mandatory on every call, and the evaluation harness runs 30 independent
    # questions that must not see each other.
    def __init__(self, supervisor=None, rag_pipeline=None, sql_pipeline=None,
                 synthesizer=None, grader=None, repairer=None, critic=None,
                 checkpointer=None, loops="repair", memory=False, **budgets):
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

        if loops:
            from src.graph.repair import SQLRepairer
            repairer = repairer or SQLRepairer()

            if loops == "all":
                from src.graph.critic import Critic
                from src.graph.grader import RetrievalGrader
                grader = grader or RetrievalGrader()
                critic = critic or Critic()

        if checkpointer is None and memory:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        self.memory = checkpointer is not None

        self.graph = build_graph(supervisor, rag_pipeline, sql_pipeline, synthesizer,
                                 grader, repairer, critic, checkpointer, **budgets)

    # ── one turn ──────────────────────────────────────────────────────────────

    @staticmethod
    def _turn_input(question: str) -> dict:
        """The input that starts a turn, including everything the last one must
        not leak into it.

        Without a checkpointer these keys are all already at their defaults and
        this is a longer way of writing `{"question": question}`. With one, the
        whole state survives, and the graph would resume mid-turn rather than
        start a new one: `passes` already at its ceiling would send the question
        straight to synthesis, and last turn's `findings` would be answered
        again. `findings` needs the explicit `None` because it accumulates -
        writing `[]` to a reducer channel appends nothing, it does not clear.

        `history` is deliberately absent: it is the one channel that carries.
        """
        return {
            "question": question,
            "findings": None,
            "resolved_question": "",
            "passes": 0, "revisions": 0, "critique": None,
            "plan": [], "awaiting": [], "agent_query": "", "mode": "parallel",
            "rag_attempts": 0, "rag_query": None, "rag_missing": "",
            "sql_attempts": 0, "sql_error": None, "last_sql": None,
            "route": "", "answer": "", "citations": [], "data": None, "sql": None,
        }

    @staticmethod
    def _config(config: dict | None, thread_id: str | None) -> dict:
        if thread_id:
            return {**(config or {}), "configurable": {
                **(config or {}).get("configurable", {}), "thread_id": thread_id}}
        return config or {}

    def run_traced(self, question: str, config: dict | None = None,
                   thread_id: str | None = None) -> dict:
        """Final state plus the sequence of nodes that produced it.

        The trajectory is what makes delegation measurable. Without it the
        harness can only score the answer, which is how routing went unevaluated
        through every earlier phase.
        """
        import time

        trajectory, state = [], {}
        t0 = time.perf_counter()
        for chunk in self.graph.stream(self._turn_input(question),
                                       self._config(config, thread_id),
                                       stream_mode="updates"):
            for node, update in chunk.items():
                trajectory.append(node)
                if isinstance(update, dict):
                    for key, value in update.items():
                        # `findings` is an accumulating channel; stream yields
                        # only this node's contribution, so append rather than
                        # overwrite or the earlier specialists disappear.
                        if key == "findings":
                            state.setdefault("findings", []).extend(value or [])
                        else:
                            state[key] = value
        state["trajectory"] = trajectory
        state["elapsed"] = time.perf_counter() - t0
        return state

    def run_state(self, question: str, config: dict | None = None,
                  thread_id: str | None = None) -> dict:
        """Full final state, including every finding and the query it answered.

        run() projects this down to v1's dict shape. Callers that need to see
        what the graph actually did - tracing, evaluation, the parity script -
        use this instead.
        """
        return self.graph.invoke(self._turn_input(question),
                                 self._config(config, thread_id))

    def run(self, question: str, config: dict | None = None,
            thread_id: str | None = None) -> dict:
        final = self.run_state(question, config, thread_id)
        return {
            "question": question,
            "route": final.get("route", "rag"),
            "answer": final.get("answer", ""),
            "citations": final.get("citations", []),
            "data": final.get("data"),
            "sql": final.get("sql"),
        }
