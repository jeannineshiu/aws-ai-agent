# src/graph/narrate.py
"""Turn a streamed node update into a line a person can read.

The supervisor's dispatch, a retry, a repair and a redraft all happen before
there is an answer to show, and until Phase 3 all of it went to stdout, where
nobody using the app could see it. The graph's whole point is that it does
something more interesting than call one pipeline; a spinner is the one display
guaranteed to hide that.

This is presentation, but it lives here rather than in app.py so it can be
tested without starting Streamlit — the mapping is where the mistakes are, and
the mistakes are silent: a wrong branch shows a plausible sentence about work
that never happened.
"""

from src.tags import ANSWER

NAMES = {"rag": "documentation", "sql": "data"}


def describe(node: str, update: dict | None, question: str = "") -> str | None:
    """One line for what just finished, or None if it is not worth saying.

    Returning None matters as much as the strings. The supervisor is woken once
    per specialist and parks the early wake-ups, and `remember` files the turn;
    narrating either would show steps the user cannot make sense of.
    """
    u = update if isinstance(update, dict) else {}

    if node == "supervisor":
        dispatched = u.get("awaiting") or []
        if not dispatched:
            return None              # a parked wake-up, or the hand-off to synthesis

        # Only shown when the supervisor actually resolved something, so the
        # user can see what "it" was taken to mean - and correct it if wrong.
        resolved = u.get("resolved_question")
        prefix = (f"Read as *{resolved}* — "
                  if resolved and question and resolved != question else "")

        joined = " + " if u.get("mode") == "parallel" else " → "
        who = joined.join(NAMES.get(a, a) for a in dispatched)

        # What each dispatched specialist was actually asked, where that is not
        # simply the question. Two cases produce one: the supervisor split a
        # two-part question, or the second half of a sequential plan is being
        # asked something the user never typed because the first half had to run
        # to produce it. Both are worth showing - a split that puts the wrong
        # clause on the wrong specialist is otherwise invisible until the number
        # comes back wrong.
        queries = u.get("agent_queries") or {}
        asked = [(a, queries[a]) for a in dispatched
                 if queries.get(a) and queries[a] != (resolved or question)]

        if len(asked) == 1 and len(dispatched) == 1:
            return f"{prefix}Asking {who}: *{asked[0][1]}*"
        if asked:
            halves = "; ".join(f"{NAMES.get(a, a)}: *{q}*" for a, q in asked)
            return f"{prefix}Splitting it — {halves}"
        return f"{prefix}Dispatching {who}"

    if node == "rag":
        if u.get("rag_query"):
            return f"Retrieval was thin — searching again for *{u['rag_query']}*"
        return "Searched the AWS documentation"

    if node == "sql":
        if u.get("sql_error"):
            return f"Query {u['sql_error']} — repairing it"
        return "Queried GitHub issues and Stack Overflow"

    if node == "synthesize":
        return "Merged both answers" if u.get("route") == "both" else "Composed the answer"

    if node == "critic":
        if u.get("critique"):
            return f"Answer outran its sources ({u['critique']}) — redrafting"
        if u.get("grounded"):
            return "Checked the answer against its sources"
        # Nothing was checked: the critic is off, or its budget is spent. Saying
        # otherwise would be the one kind of wrong line that reads as right.
        return None

    return None                      # remember, and anything added after it


# ── which tokens are the answer ───────────────────────────────────────────────
#
# A turn makes several model calls and the app types out exactly one of them.
# Picking it needs two facts, and both are on the stream already: whether this
# turn is going to end in a merge, and which call produced the token.

def expects_a_merge(update: dict | None) -> bool | None:
    """Will this dispatch end with the synthesizer writing the answer?

    None when the update is not a dispatch — a parked wake-up or the hand-off to
    synthesis — so a caller can keep looking rather than take silence for a no.

    Two specialists means `synthesize` merges their findings, and the merge is
    the answer; one means the specialist's own prose is passed through untouched
    and *that* is the answer. The count is `awaiting` plus `plan` because a
    sequential plan dispatches its two specialists one at a time: at the first
    dispatch only one of them is being awaited, and the other is still queued.
    """
    if not isinstance(update, dict):
        return None
    awaiting = update.get("awaiting") or []
    if not awaiting:
        return None
    return len(awaiting) + len(update.get("plan") or []) > 1


def carries_the_answer(node: str, tags, merge_expected: bool) -> bool:
    """Is a token from this call part of the answer the user will read?

    The tag alone is not enough. On a two-specialist turn both specialists write
    prose and both are tagged, but neither is what the user gets - the merge
    rewrites them into one answer, and typing all three out in turn would show
    two paragraphs that are then replaced.
    """
    if ANSWER not in (tags or ()):
        return False
    return not (merge_expected and node != "synthesize")
