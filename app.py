# app.py
import os
import json
import glob
import uuid
import streamlit as st
from src.agent.agent import AWSAgent

# Page config
st.set_page_config(
    page_title="AWS AI/ML Agent",
    page_icon="🤖",
    layout="wide",
)

# Which implementation to run. The graph is the default from Phase 3: the
# evaluation says it answers better (relevancy 0.487 -> 0.724, adversarial SQL
# 60% -> 100%), and it is the only one of the two that can hold a conversation.
# Set AGENT_IMPL=v1 to get the linear pipeline back for comparison.
AGENT_IMPL = os.getenv("AGENT_IMPL", "graph").lower()
CONFIRM_SQL = os.getenv("CONFIRM_SQL", "1") not in ("0", "false", "no")


# Initialize agent (cached so it only loads once)
@st.cache_resource
def load_agent(impl: str):
    if impl == "graph":
        from src.graph.builder import GraphAgent
        # memory=True compiles a checkpointer, which is what makes a follow-up
        # question a follow-up rather than a fresh one.
        # confirm_sql stops a query the reviewer cannot classify, or one the
        # repairer wrote rather than the question, and puts it to the person
        # sitting here. The app is the only place with someone to ask.
        return GraphAgent(memory=True, confirm_sql=CONFIRM_SQL)
    return AWSAgent()

agent = load_agent(AGENT_IMPL)
IS_GRAPH = AGENT_IMPL == "graph"

if IS_GRAPH:
    from src.graph.builder import project
    from src.graph.narrate import describe




# --- Sidebar ---
with st.sidebar:
    st.title("🤖 AWS AI/ML Agent")
    st.caption("Ask anything about AWS AI/ML services")

    st.divider()
    st.markdown("**Example questions**")

    example_questions = {
        "📄 Documentation": [
            "What is Amazon Bedrock?",
            "How does SageMaker Model Monitor work?",
            "When should I use Rekognition vs Comprehend?",
        ],
        "📊 Data Analysis": [
            "Which AWS service has the most unanswered questions?",
            "How many SageMaker questions were asked in 2023?",
            "Which repo has the most open GitHub issues?",
        ],
        "🔀 Combined": [
            "What are the most common SageMaker issues and how does training work?",
            "Which service has the most questions and what does it do?",
        ],
    }

    for category, questions in example_questions.items():
        st.markdown(f"**{category}**")
        for q in questions:
            if st.button(q, key=q, width="stretch"):
                st.session_state.selected_question = q

    st.divider()
    if IS_GRAPH and st.button("🔄 New conversation", width="stretch"):
        # A thread is a conversation. Dropping the transcript without dropping
        # the thread would leave the agent resolving "it" against turns the
        # user can no longer see.
        st.session_state.messages = []
        for key in ("thread_id", "awaiting_sql", "resume_with"):
            st.session_state.pop(key, None)
        st.rerun()

    st.caption("Built with LangGraph · ChromaDB · OpenAI · SQLite")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Chat
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.title("AWS AI/ML Knowledge & Analytics Agent")
    st.caption("Combines AWS documentation (RAG) with developer data analysis (SQL)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # The agent is @st.cache_resource, so one checkpointer is shared by every
    # browser session. The thread id is what keeps those conversations apart.
    if IS_GRAPH and "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("data") is not None:
                st.dataframe(msg["data"], width="stretch")
            if msg.get("sql"):
                with st.expander("View SQL query"):
                    st.code(msg["sql"], language="sql")
            if msg.get("citations"):
                with st.expander(f"View sources ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.markdown(f"- [{c['title']}]({c['url']}) — `{c['service']}`")
            if msg.get("steps"):
                with st.expander(f"How this was answered ({len(msg['steps'])} steps)"):
                    for line in msg["steps"]:
                        st.markdown(f"- {line}")

    if "selected_question" in st.session_state:
        question = st.session_state.pop("selected_question")
        st.session_state.pending_question = question

    user_input = st.chat_input("Ask about AWS AI/ML services...")
    if user_input:
        st.session_state.pending_question = user_input

    # A turn parked at the confirmation gate, rendered on its own rerun: a
    # button click is only visible to the script run after the one that drew it.
    if "awaiting_sql" in st.session_state and "resume_with" not in st.session_state:
        held = st.session_state.awaiting_sql
        with st.chat_message("assistant"):
            st.warning(f"Before I run this — {held['payload']['reason']}.")
            st.code(held["payload"]["sql"], language="sql")
            st.caption(f"Written to answer: *{held['payload']['question']}*")
            run_it, skip_it = st.columns(2)
            if run_it.button("Run it", type="primary", width="stretch"):
                st.session_state.resume_with = True
                st.rerun()
            if skip_it.button("Answer without it", width="stretch"):
                st.session_state.resume_with = False
                st.rerun()
        st.stop()

    if "pending_question" in st.session_state or "resume_with" in st.session_state:
        resuming = "resume_with" in st.session_state

        if resuming:
            held = st.session_state.pop("awaiting_sql")
            question, steps = held["question"], list(held["steps"])
            decision = st.session_state.pop("resume_with")
            steps.append("Ran the held query" if decision
                         else "Skipped the held query")
            events = agent.stream_answer(resume=decision,
                                         thread_id=st.session_state.thread_id)
        else:
            question, steps = st.session_state.pop("pending_question"), []
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})
            events = None

        with st.chat_message("assistant"):
            status = None
            try:
                if IS_GRAPH:
                    thread_id = st.session_state.thread_id
                    if events is None:
                        # Narrate the turn as it happens. The interesting
                        # decisions - who was dispatched, what was retried, what
                        # was sent back to be redrafted - are all over before the
                        # answer exists, and a spinner shows none of them.
                        events = agent.stream_answer(question, thread_id=thread_id)

                    # Held by name rather than entered with `with`, because the
                    # answer placeholder has to be created after it to render
                    # below it, and then written to from inside the same loop.
                    # The answer starts arriving while the steps are still coming.
                    status = st.status("Working…", expanded=True)
                    for line in steps:
                        status.markdown(f"- {line}")
                    route_box, answer_box = st.empty(), st.empty()

                    draft = ""
                    for kind, payload in events:
                        if kind == "node":
                            node, update = payload
                            if line := describe(node, update, question):
                                steps.append(line)
                                status.markdown(f"- {line}")
                        elif kind == "restart":
                            # The critic rejected the draft. What follows is a
                            # different answer to the same question, not more of
                            # this one.
                            draft = ""
                        else:
                            draft += payload
                            answer_box.markdown(draft)

                    held = agent.pending_confirmation(thread_id)
                    if held:
                        status.update(label="Waiting for you",
                                      state="complete", expanded=False)
                        st.session_state.awaiting_sql = {
                            "question": question, "payload": held, "steps": steps}
                        st.rerun()

                    status.update(label=f"Answered in {len(steps)} steps",
                                  state="complete", expanded=False)

                    # The turn was streamed for the display above; the answer
                    # itself comes from the checkpointer, already assembled.
                    # It is not always the string that was typed out: synthesis
                    # strips citations that point at nothing, and that happens
                    # after the last token. Rewriting the placeholder below is
                    # what applies it.
                    result = project(question, agent.state_of(thread_id))
                else:
                    with st.spinner("Thinking..."):
                        result = agent.run(question)
                    route_box, answer_box = st.empty(), st.empty()
            except Exception as e:
                if status is not None:
                    # Nothing closes the box on the way out of a plain `try`.
                    status.update(label="Failed", state="error", expanded=False)
                st.error(f"Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                    "data": None, "sql": None, "citations": [], "steps": steps,
                })
                st.stop()

            route_colors = {"rag": "🟢", "sql": "🔵", "both": "🟣"}
            route_emoji = route_colors.get(result["route"], "⚪")
            route_box.caption(f"{route_emoji} Route: `{result['route'].upper()}`")
            answer_box.markdown(result["answer"])

            if result.get("data") is not None and not result["data"].empty:
                st.dataframe(result["data"], width="stretch")
            if result.get("sql"):
                with st.expander("View SQL query"):
                    st.code(result["sql"], language="sql")
            if result.get("citations"):
                with st.expander(f"View sources ({len(result['citations'])})"):
                    for c in result["citations"]:
                        st.markdown(f"- [{c['title']}]({c['url']}) — `{c['service']}`")
            if steps:
                with st.expander(f"How this was answered ({len(steps)} steps)"):
                    for line in steps:
                        st.markdown(f"- {line}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "data": result.get("data"),
            "sql": result.get("sql"),
            "citations": result.get("citations", []),
            "steps": steps,
        })


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Evaluation Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_eval:
    st.title("Evaluation Dashboard")
    st.caption("RAGAS metrics for RAG · exact-match accuracy for SQL · run via `python scripts/run_evaluation.py`")

    # Load latest report
    report_files = sorted(glob.glob("data/processed/eval_report_*.json"), reverse=True)

    if not report_files:
        st.info("No evaluation report found. Run `python scripts/run_evaluation.py` to generate one.")
        st.stop()

    latest = report_files[0]
    with open(latest) as f:
        report = json.load(f)

    st.caption(f"Report: `{os.path.basename(latest)}` · generated {report['timestamp']}")

    st.divider()

    # ── RAG Metrics ───────────────────────────────────────────────────────────
    st.subheader("RAG Metrics (RAGAS)")
    st.caption("Evaluated on 10 ground-truth Q&A pairs using `gpt-4o-mini`")

    rag = report["rag_metrics"]
    metric_labels = {
        "faithfulness":      ("Faithfulness",       "Answer only uses information from retrieved context"),
        "answer_relevancy":  ("Answer Relevancy",   "Answer directly addresses the question"),
        "context_precision": ("Context Precision",  "Retrieved chunks are relevant to the question"),
        "context_recall":    ("Context Recall",     "Retrieved chunks cover the ground-truth answer"),
    }

    cols = st.columns(4)
    for col, (key, (label, tooltip)) in zip(cols, metric_labels.items()):
        score = rag.get(key)
        with col:
            if score is None or (isinstance(score, float) and score != score):
                st.metric(label=label, value="N/A", help=tooltip)
            else:
                delta_color = "normal" if score >= 0.7 else "inverse"
                st.metric(label=label, value=f"{score:.3f}", help=tooltip,
                          delta=f"{'✓ Good' if score >= 0.7 else '⚠ Low'}",
                          delta_color=delta_color)

    # Bar chart
    import pandas as pd
    chart_data = {
        label: [rag.get(key, 0) or 0]
        for key, (label, _) in metric_labels.items()
    }
    chart_df = pd.DataFrame(chart_data)
    st.bar_chart(chart_df, height=220)

    st.divider()

    # ── SQL Metrics ───────────────────────────────────────────────────────────
    st.subheader("SQL Accuracy")
    st.caption("Generated SQL vs ground-truth SQL, numeric results compared with 5% tolerance")

    sql_m = report["sql_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{sql_m['accuracy']:.0%}")
    c2.metric("Correct", f"{sql_m['correct_queries']} / {sql_m['total_queries']}")
    c3.metric("Total queries", sql_m["total_queries"])

    st.markdown("**Per-question breakdown**")
    rows = []
    for row in report["sql_details"]:
        rows.append({
            "Result": "✅" if row["results_match"] else "❌",
            "Question": row["question"],
            "Expected": str(row["ground_truth_value"]) if row.get("ground_truth_value") is not None else "",
            "Got": str(row["generated_value"]) if row.get("generated_value") is not None else "",
            "Generated SQL": row.get("generated_sql", ""),
        })
    detail_df = pd.DataFrame(rows)
    st.dataframe(detail_df, width="stretch", hide_index=True)
