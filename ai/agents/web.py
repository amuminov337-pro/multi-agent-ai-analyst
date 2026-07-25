"""Web agent — Tavily search for questions outside the document corpus (F4).

Rubric requirement: when TAVILY_API_KEY is absent the agent must skip
GRACEFULLY — no exception, no crash — so the whole graph still runs on a
Gemini-only setup. Network and quota errors are handled the same way: the
step is recorded with the reason and the run continues.

Retrieved hits are appended to state["documents"] (shared with the F3
retriever) with their source URL inlined, so the critic (F8) and the
frontend (F13) can show where each claim came from.
"""

from __future__ import annotations

from typing import List, Optional

from ai.agents.retriever import merge_documents
from ai.config import get_settings
from ai.state import AgentState, push_step

WEB_MAX_RESULTS = 4


def web_search(question: str, max_results: int = WEB_MAX_RESULTS) -> List[str]:
    """Run a Tavily search and return formatted chunks.

    Returns an empty list when Tavily is not configured — callers treat
    "no key" and "no hits" the same way. Raises only on genuinely
    unexpected client errors, which web_agent() catches.
    """
    settings = get_settings()
    if not settings.tavily_enabled:
        return []

    from tavily import TavilyClient  # imported lazily: optional dependency

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(question, max_results=max_results)

    chunks: List[str] = []
    for hit in response.get("results", []):
        title = (hit.get("title") or "").strip()
        url = (hit.get("url") or "").strip()
        content = (hit.get("content") or "").strip()
        if not content:
            continue
        chunks.append(f"[web] {title} ({url})\n{content}")
    return chunks


def web_agent(state: AgentState) -> dict:
    """LangGraph node: question -> live web results in state.

    Never raises. Three outcomes, all recorded in state["steps"]:
      * skipped  — no TAVILY_API_KEY configured
      * N hits   — search succeeded
      * failed   — network/quota error, run continues without web evidence
    """
    settings = get_settings()

    if not settings.tavily_enabled:
        return {
            "documents": state.get("documents") or [],
            "steps": push_step(state, "web(skipped: no TAVILY_API_KEY)"),
        }

    try:
        chunks = web_search(state["question"])
        label = f"web({len(chunks)} hits)"
    except Exception as exc:  # quota, network, client change — never fatal
        chunks = []
        label = f"web(failed: {type(exc).__name__})"

    return {
        "documents": merge_documents(state.get("documents") or [], chunks),
        "steps": push_step(state, label),
    }