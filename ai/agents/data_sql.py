"""Data agent — text-to-SQL over the analytics database (F5).

The agent turns a plain-language question into ONE SQLite SELECT, runs it
through the read-only layer in ai/db.py, and writes both the query and its
result into state["sql_result"] — the query is kept deliberately, so the
critic (F8) and the frontend (F13) can show HOW the number was obtained.

Safety is not implemented here: every query goes through db.run_query(),
which enforces no-comments, single-statement, SELECT-only, no-forbidden-
keywords, and an OS-level read-only connection. This node only handles
prompting and error reporting, and never raises — a rejected or broken
query is recorded as evidence so the supervisor can re-route instead of
the graph crashing.
"""

from __future__ import annotations

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from ai import db
from ai.config import get_settings
from ai.state import AgentState, push_step

SQL_PROMPT = """You are a SQLite expert. Write ONE query that answers the question.

Database schema with sample rows:
{schema}

Question: {question}

Rules:
- Output ONLY the SQL query. No explanation, no markdown fences, no comments.
- It must be a single SELECT statement. Never INSERT, UPDATE, DELETE, DROP,
  ALTER or CREATE anything — the connection is read-only and such a query
  will be rejected.
- Never use SQL comments (-- or /* */); a query containing them is rejected.
- Use exact column names from the schema above.
- Prefer an aggregate (COUNT, SUM, AVG, ROUND) when the question asks for a
  number, so the answer is a single value.
"""


def _llm() -> ChatGoogleGenerativeAI:
    """Chat model for SQL generation.

    No temperature is set: gemini-3.6-flash uses fixed sampling defaults
    and warns on every call when the parameter is passed.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(model=settings.gemini_model)


def _extract_text(response: Any) -> str:
    """Get plain text out of a LangChain response.

    Newer Gemini models return `content` as a LIST of content blocks
    (e.g. [{"type": "text", "text": "SELECT ..."}]) rather than a string.
    Stringifying that list feeds Python's repr to the SQL guard instead of
    the query, so every block is unpacked explicitly here.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    return str(content)


def generate_sql(question: str, schema: str) -> str:
    """Ask the LLM for one SQL query (raw text, not yet validated)."""
    prompt = SQL_PROMPT.format(schema=schema, question=question)
    return _extract_text(_llm().invoke(prompt))


def answer_with_sql(question: str) -> str:
    """Full text-to-SQL round trip. Returns a formatted evidence string."""
    schema = db.get_schema()
    raw_sql = generate_sql(question, schema)
    query, rows, columns = db.run_query(raw_sql)
    result = db.format_result(rows, columns)
    return f"SQL: {query}\nResult:\n{result}"


def data_agent(state: AgentState) -> dict:
    """LangGraph node: question -> SQL query + result in state["sql_result"].

    Never raises. Failure modes recorded in steps:
      * data(rejected)    — the safety layer blocked the generated query
      * data(sql error)   — the query was safe but invalid SQL
      * data(no database) — the DB has not been seeded
      * data(failed: X)   — anything else (API/network)
    """
    question = state["question"]

    try:
        evidence = answer_with_sql(question)
        label = "data(sql)"
    except db.UnsafeQueryError as exc:
        evidence = f"SQL rejected by the read-only guard: {exc}"
        label = "data(rejected)"
    except db.DatabaseMissingError as exc:
        evidence = f"Database unavailable: {exc}"
        label = "data(no database)"
    except Exception as exc:
        evidence = f"SQL execution failed ({type(exc).__name__}): {exc}"
        label = f"data(failed: {type(exc).__name__})"

    return {
        "sql_result": evidence,
        "steps": push_step(state, label),
    }