# src/tags.py
"""Tags on the model calls a turn makes, so a caller can tell them apart.

One turn makes several calls and only one of them writes the prose the user
reads. The rest — planning the dispatch, writing the SQL, grading retrieval —
are working notes. LangGraph's `messages` stream carries the tags of the call
that produced each token, so a UI that wants to type the answer out as it is
written needs some way to know which tokens are the answer; without one it
would type a JSON plan into the answer box and then replace it.

It lives at the top of `src` rather than under `src/graph` because both halves
need it and the dependency only runs one way: the pipelines set the tag, the
graph reads it back off the stream.
"""

ANSWER = "answer"
