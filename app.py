# app.py
import os
import json
import glob
import streamlit as st
from src.agent.agent import AWSAgent

# Page config
st.set_page_config(
    page_title="AWS AI/ML Agent",
    page_icon="🤖",
    layout="wide",
)

# Which implementation to run. v1 (AWSAgent) stays the default through Phase 0 —
# the graph is proven equivalent, not yet an improvement, so switching by default
# would add risk for no gain. Set AGENT_IMPL=graph to exercise the LangGraph port.
AGENT_IMPL = os.getenv("AGENT_IMPL", "v1").lower()


# Initialize agent (cached so it only loads once)
@st.cache_resource
def load_agent(impl: str):
    if impl == "graph":
        from src.graph.builder import GraphAgent
        return GraphAgent()
    return AWSAgent()

agent = load_agent(AGENT_IMPL)

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
    st.caption("Built with LangChain · ChromaDB · OpenAI · SQLite")


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

    if "selected_question" in st.session_state:
        question = st.session_state.pop("selected_question")
        st.session_state.pending_question = question

    user_input = st.chat_input("Ask about AWS AI/ML services...")
    if user_input:
        st.session_state.pending_question = user_input

    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = agent.run(question)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {e}",
                        "data": None, "sql": None, "citations": [],
                    })
                    st.stop()

            route_colors = {"rag": "🟢", "sql": "🔵", "both": "🟣"}
            route_emoji = route_colors.get(result["route"], "⚪")
            st.caption(f"{route_emoji} Route: `{result['route'].upper()}`")
            st.markdown(result["answer"])

            if result.get("data") is not None and not result["data"].empty:
                st.dataframe(result["data"], width="stretch")
            if result.get("sql"):
                with st.expander("View SQL query"):
                    st.code(result["sql"], language="sql")
            if result.get("citations"):
                with st.expander(f"View sources ({len(result['citations'])})"):
                    for c in result["citations"]:
                        st.markdown(f"- [{c['title']}]({c['url']}) — `{c['service']}`")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "data": result.get("data"),
            "sql": result.get("sql"),
            "citations": result.get("citations", []),
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
