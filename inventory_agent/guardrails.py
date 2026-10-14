"""Checks that every SQL statement must pass before it touches the database.

Defence in depth: the database connection is already read-only, and this module also
rejects anything that is not a single SELECT statement.
"""
from __future__ import annotations

import re
import sqlite3
import time

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|"
    r"TRUNCATE|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|LOAD_EXTENSION)\b",
    re.IGNORECASE,
)


class GuardrailError(ValueError):
    """Raised when a query is not allowed. The message is safe to show to the model."""


def validate_sql(sql: str) -> str:
    """Return the cleaned SQL, or raise GuardrailError."""
    if not isinstance(sql, str) or not sql.strip():
        raise GuardrailError("Empty query.")
    cleaned = sql.strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    if ";" in cleaned:
        raise GuardrailError("Only one statement is allowed.")
    if "--" in cleaned or "/*" in cleaned:
        raise GuardrailError("SQL comments are not allowed.")
    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        raise GuardrailError("Only SELECT queries are allowed.")
    match = FORBIDDEN.search(cleaned)
    if match:
        raise GuardrailError(f"The keyword {match.group(1).upper()} is not allowed. Only read-only SELECT queries are allowed.")
    if re.search(r"sqlite_", cleaned, re.IGNORECASE):
        raise GuardrailError("System tables are not accessible.")
    return cleaned


def set_timeout(con: sqlite3.Connection, seconds: float) -> None:
    """Stop any statement on this connection that runs longer than `seconds`."""
    deadline = time.monotonic() + seconds

    def check() -> int:
        return 1 if time.monotonic() > deadline else 0  # non-zero interrupts the query

    con.set_progress_handler(check, 10_000)


def clear_timeout(con: sqlite3.Connection) -> None:
    con.set_progress_handler(None, 0)
