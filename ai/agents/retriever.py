"""Retriever agent — RAG over the ingested document corpus (F3).

Reads the question from AgentState, pulls the top-k most similar chunks
out of Qdrant (populated in F2), and appends them to state["documents"].

Design notes:
  * Documents are APPENDED, not replaced: the web agent (F4) writes into
    the same list, and the supervisor (F7) may call several agents for
    one question. Duplicates are removed while preserving order.
  * An empty or missing collection is not an error. The agent returns no
    documents and records the step, so the graph keeps running instead of
    crashing mid-run.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from ai.config import get_settings
from ai.state import AgentState, push_step
from ai.vectorstore import get_vectorstore


def merge_documents(existing: List[str], new: List[str]) -> List[str]:
    """Merge two chunk lists, keeping order and dropping repeats.

    Shared by every agent that writes into state["documents"] (F3 retriever,
    F4 web) so evidence accumulates instead of overwriting.
    """
    merged: List[str] = []
    seen = set()
    for text in list(existing) + list(new):
        key = text.strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(text)
    return merged


def retrieve(question: str, k: Optional[int] = None) -> List[Document]:
    """Plain similarity search — reusable by F10 memory and F11 eval."""
    settings = get_settings()
    top_k = k if k is not None else settings.retriever_k
    store = get_vectorstore()
    return store.similarity_search(question, k=top_k)


def retriever_agent(state: AgentState) -> dict:
    """LangGraph node: question -> top-k document chunks in state."""
    question = state["question"]

    try:
        hits = retrieve(question)
        texts = [doc.page_content for doc in hits]
        label = f"retriever({len(texts)} chunks)"
    except Exception as exc:  # never take the whole graph down
        texts = []
        label = f"retriever(failed: {type(exc).__name__})"

    return {
        "documents": merge_documents(state.get("documents") or [], texts),
        "steps": push_step(state, label),
    }