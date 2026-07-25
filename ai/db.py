"""Read-only SQLite access layer for the data agent (F5).

Rubric requirement: "read-only DB user; reject any query that isn't a
SELECT. Never let it run DROP/DELETE." Model-written SQL is untrusted
input, so this module applies FOUR independent layers of defence — any one
of them alone would be a single point of failure:

  1. No comments        — a query containing `--`, `/*` or `*/` is rejected
     outright. We deliberately REJECT rather than SANITIZE: stripping a
     comment and running the remainder means trusting our own stripper,
     and an attacker only has to out-think the stripper once.
  2. Single statement   — a `;` inside the query is rejected, so a stacked
     "SELECT ...; DROP ..." can never reach SQLite.
  3. Read statement     — the query must start with SELECT or WITH, and no
     write/DDL/engine keyword (INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH,
     PRAGMA, ...) may appear anywhere. The scan is word-boundary based, so
     harmless column names like `updated_at` are not false positives.
  4. OS-level read-only — SQLite is opened through the URI
     `file:<path>?mode=ro`. Even if a write somehow passed layers 1–3, the
     database file physically cannot be modified through this handle.

Results are also row-capped so a runaway query can't flood the context.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import List, Tuple

from ai.config import get_settings

MAX_ROWS = 50
SAMPLE_ROWS = 3

# Everything that could modify data, schema, or the engine's environment.
FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "grant",
    "revoke",
    "begin",
    "commit",
    "rollback",
    "savepoint",
    "analyze",
)

_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE
)
# ```sql ... ``` or ``` ... ``` — LLMs habitually wrap SQL in fences.
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_COMMENT_MARKER_RE = re.compile(r"--|/\*|\*/")


class UnsafeQueryError(ValueError):
    """Raised when model-written SQL is not a safe read-only statement."""


class DatabaseMissingError(FileNotFoundError):
    """Raised when the SQLite file has not been seeded yet."""


def extract_sql(raw: str) -> str:
    """Pull the SQL candidate out of an LLM response.

    This is FORMATTING only, never security: whatever comes out still has
    to survive every check in assert_read_only(). It handles the one thing
    models reliably do wrong — wrapping the query in a markdown fence — and
    drops a single trailing semicolon.
    """
    text = (raw or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    return text.strip().rstrip(";").strip()


def assert_read_only(sql: str) -> str:
    """Validate model-written SQL. Returns the cleaned query or raises.

    Layers 1–3. Deliberately strict: anything it is not sure about is
    rejected rather than executed.
    """
    query = extract_sql(sql)

    if not query:
        raise UnsafeQueryError("Empty query.")

    # Layer 1: no comments — reject, never sanitize.
    marker = _COMMENT_MARKER_RE.search(query)
    if marker:
        raise UnsafeQueryError(
            f"SQL comments are not allowed (found '{marker.group(0)}') — "
            "comments are a common way to smuggle a second statement."
        )

    # Layer 2: single statement only — no stacked "SELECT ...; DROP ..."
    if ";" in query:
        raise UnsafeQueryError(
            "Multiple statements are not allowed (found ';' inside the query)."
        )

    # Layer 3a: must be a read statement
    first_word = query.split(None, 1)[0].lower()
    if first_word not in ("select", "with"):
        raise UnsafeQueryError(
            f"Only SELECT queries are allowed, got '{first_word.upper()}'."
        )

    # Layer 3b: no write/DDL/engine keywords anywhere in the query
    match = _FORBIDDEN_RE.search(query)
    if match:
        raise UnsafeQueryError(
            f"Forbidden keyword '{match.group(1).upper()}' in query — "
            "the data agent is read-only."
        )

    return query


def get_connection() -> sqlite3.Connection:
    """Open the database READ-ONLY at the OS level (layer 4)."""
    settings = get_settings()
    path: Path = settings.sqlite_path
    if not path.exists():
        raise DatabaseMissingError(
            f"SQLite database not found at {path}. "
            "Run `python scripts/seed_db.py` first."
        )
    uri = f"file:{path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def get_schema() -> str:
    """Human-readable schema + sample rows, used to prompt the LLM."""
    conn = get_connection()
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]

        blocks: List[str] = []
        for table in tables:
            cols = conn.execute(f"SELECT * FROM {table} LIMIT 0").description
            col_names = [c[0] for c in cols]
            header = f"TABLE {table}({', '.join(col_names)})"

            rows = conn.execute(
                f"SELECT * FROM {table} LIMIT {SAMPLE_ROWS}"
            ).fetchall()
            samples = "\n".join("  " + " | ".join(str(v) for v in r) for r in rows)
            blocks.append(f"{header}\n  -- sample rows --\n{samples}")
        return "\n\n".join(blocks)
    finally:
        conn.close()


def run_query(sql: str) -> Tuple[str, List[tuple], List[str]]:
    """Validate and execute a read-only query.

    Returns (cleaned_sql, rows, column_names). Raises UnsafeQueryError if
    the safety layers reject it, sqlite3.Error if the SQL is invalid.
    """
    query = assert_read_only(sql)
    conn = get_connection()
    try:
        cursor = conn.execute(query)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(MAX_ROWS)
        return query, rows, columns
    finally:
        conn.close()


def format_result(rows: List[tuple], columns: List[str]) -> str:
    """Render query output compactly for the agent state and the critic."""
    if not rows:
        return "(no rows)"
    header = " | ".join(columns) if columns else ""
    body = "\n".join(" | ".join("" if v is None else str(v) for v in r) for r in rows)
    capped = f"\n(showing first {MAX_ROWS} rows)" if len(rows) == MAX_ROWS else ""
    return (f"{header}\n{body}{capped}" if header else f"{body}{capped}")