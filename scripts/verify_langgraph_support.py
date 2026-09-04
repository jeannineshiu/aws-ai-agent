# scripts/verify_langgraph_support.py
"""Verify that LangGraph provides every capability the multi-agent plan needs
on the pinned langchain-core 0.3.x line.

Context: requirements.txt holds langchain-core below 1.0 because ragas 0.4.3
hard-imports langchain_community.chat_models.vertexai, a module removed in the
langchain-community 0.4.x line that core 1.x requires. This script exists to
prove the pin costs us nothing — if it passes, the migration needs no bump.

Pure graph mechanics; makes no API calls and costs nothing to run.

    python scripts/verify_langgraph_support.py
"""
import operator
from typing import Annotated, TypedDict, Literal

import langchain_core, langgraph
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

print(f"langchain-core {langchain_core.__version__}")

ok = []
def check(name, cond, detail=""):
    ok.append((name, cond, detail))

class State(TypedDict):
    question: str
    findings: Annotated[list[str], operator.add]   # parallel-safe reducer
    attempts: int
    answer: str

# ── 1. supervisor returning Command(goto=[...]) → parallel fan-out ──
def supervisor(state: State) -> Command[Literal["docs", "analytics"]]:
    return Command(goto=["docs", "analytics"], update={"attempts": state["attempts"]})

def docs(state: State):      return {"findings": ["docs:hit"]}
def analytics(state: State): return {"findings": ["sql:hit"]}

# ── 2. cycle with a budget ──
def critic(state: State):
    if state["attempts"] < 2:
        return {"attempts": state["attempts"] + 1}
    return {"answer": "final"}

def route_after_critic(state: State) -> Literal["supervisor", "__end__"]:
    return END if state.get("answer") else "supervisor"

g = StateGraph(State)
g.add_node("supervisor", supervisor)
g.add_node("docs", docs)
g.add_node("analytics", analytics)
g.add_node("critic", critic)
g.add_edge(START, "supervisor")
g.add_edge("docs", "critic")
g.add_edge("analytics", "critic")
g.add_conditional_edges("critic", route_after_critic)

saver = InMemorySaver()
app = g.compile(checkpointer=saver)

cfg = {"configurable": {"thread_id": "t1"}}
out = app.invoke({"question": "q", "findings": [], "attempts": 0, "answer": ""}, cfg)

check("StateGraph + conditional cycle", out["answer"] == "final", f"attempts={out['attempts']}")
check("operator.add reducer merges parallel writes",
      sorted(set(out["findings"])) == ["docs:hit", "sql:hit"],
      f"findings={out['findings']}")
check("Command(goto=[...]) fan-out", len(out["findings"]) >= 2)

# ── 3. checkpointer keeps thread state across invocations ──
snap = app.get_state(cfg)
check("checkpointer persists state", snap.values["answer"] == "final")
hist = list(app.get_state_history(cfg))
check("time travel / get_state_history", len(hist) > 1, f"{len(hist)} checkpoints")

# ── 4. streaming per-node updates ──
seen = [list(ch.keys())[0] for ch in app.stream(
    {"question": "q", "findings": [], "attempts": 0, "answer": ""},
    {"configurable": {"thread_id": "t2"}}, stream_mode="updates")]
check("stream_mode='updates' emits per-node", "supervisor" in seen and "critic" in seen,
      f"{len(seen)} updates")

# ── 5. interrupt() → human-in-the-loop ──
class S2(TypedDict):
    sql: str
    approved: str

def gen(state: S2):      return {"sql": "SELECT 1"}
def approve(state: S2):  return {"approved": interrupt({"sql": state["sql"]})}

g2 = StateGraph(S2)
g2.add_node("gen", gen); g2.add_node("approve", approve)
g2.add_edge(START, "gen"); g2.add_edge("gen", "approve"); g2.add_edge("approve", END)
app2 = g2.compile(checkpointer=InMemorySaver())

c2 = {"configurable": {"thread_id": "hitl"}}
r = app2.invoke({"sql": "", "approved": ""}, c2)
paused = "__interrupt__" in r
check("interrupt() pauses the graph", paused, str(list(r.keys())))

resumed = app2.invoke(Command(resume="yes"), c2)
check("Command(resume=...) continues", resumed.get("approved") == "yes")

# ── report ──
print()
for name, cond, detail in ok:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
print(f"\n{sum(1 for _,c,_ in ok if c)}/{len(ok)} passed")
