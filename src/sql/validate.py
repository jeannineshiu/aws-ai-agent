# src/sql/validate.py
"""Decide whether a generated query may run.

The check this replaces was one regex sweep for six keywords over the raw query
text, and a binary answer. It was wrong in both directions, and both directions
were measured before this module was written:

    SELECT COUNT(*) ... WHERE title LIKE '%delete endpoint%'   rejected
    SELECT repo, COUNT(*) ... WHERE body LIKE '%create model%'  rejected
    WITH t AS (SELECT ...) SELECT * FROM t                      rejected
    SELECT * FROM issues LIMIT 50; SELECT * FROM stackoverflow  allowed

The first two are the app's own subject matter - people ask about deleting
endpoints and creating models - and the deny-list could not tell a keyword from
the same letters inside a string literal. The third is ordinary SQL. The fourth
is two statements, which the scan had no concept of.

So: strip comments and literals first, then judge what is left, and split the
answer three ways. `reject` is for what must not run. `allow` is for a single
read-only statement. `confirm` is the one the old check could not express - the
query is not obviously dangerous and not confidently readable either, and the
honest response to that is to ask rather than to guess in either direction.
README's SQL section says this is not a security firewall; a middle tier is
what that admission looks like in code.
"""
import re
from typing import Literal, NamedTuple

# Statement keywords that change data or reach outside the query. Matched on the
# literal-stripped text, so a question *about* deleting an endpoint is unaffected.
# `REPLACE` is deliberately absent: it is also SQLite's string function, and
# `replace(title, 'a', 'b')` is a read.
FORBIDDEN = ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE",
             "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ANALYZE")

READ_ONLY_STARTS = ("SELECT", "WITH")


class Review(NamedTuple):
    verdict: Literal["allow", "confirm", "reject"]
    reason: str


def strip_literals(sql: str) -> str | None:
    """Replace string literals and comments with blanks, preserving length.

    Returns None when the quoting does not close, which is not a rejection: it
    means this function cannot see the query clearly, and something that cannot
    be read should not be waved through by the same code that cannot read it.
    """
    out, i, n = [], 0, len(sql)
    while i < n:
        c = sql[i]
        if c == "-" and sql[i:i + 2] == "--":
            end = sql.find("\n", i)
            end = n if end == -1 else end
            out.append(" " * (end - i)); i = end
        elif c == "/" and sql[i:i + 2] == "/*":
            end = sql.find("*/", i + 2)
            if end == -1:
                return None
            out.append(" " * (end + 2 - i)); i = end + 2
        elif c in "'\"":
            j = i + 1
            while j < n:
                if sql[j] == c:
                    if sql[j:j + 2] == c * 2:   # '' escapes a quote inside a literal
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                return None
            out.append(" " * (j + 1 - i)); i = j + 1
        else:
            out.append(c); i += 1
    return "".join(out)


def review(sql: str) -> Review:
    """Classify a generated query as allow / confirm / reject."""
    if not sql or not sql.strip():
        return Review("reject", "the query is empty")

    stripped = strip_literals(sql)
    if stripped is None:
        return Review("confirm", "the quoting does not close, so it cannot be read")

    statements = [s for s in stripped.split(";") if s.strip()]
    if len(statements) > 1:
        # Not merely unsupported by the driver: one statement was reviewed and a
        # second was going to run behind it.
        return Review("reject", f"there are {len(statements)} statements, not one")

    body = statements[0] if statements else ""
    upper = body.upper()

    for keyword in FORBIDDEN:
        if re.search(rf"\b{keyword}\b", upper):
            return Review("reject", f"it contains {keyword}")

    first = upper.strip().split()[0] if upper.strip() else ""
    if first in READ_ONLY_STARTS:
        return Review("allow", "a single read-only statement")

    return Review("confirm", f"it starts with {first or 'nothing recognisable'}, "
                             "which is neither a known read nor a known write")
