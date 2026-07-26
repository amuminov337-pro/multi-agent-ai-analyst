"""Shared Gemini LLM access for every agent.

One place that knows how to build the chat model and how to get plain text
out of its response, so the SQL agent (F5), code agent (F6), supervisor
(F7) and critic (F8) never duplicate that logic.
"""

from __future__ import annotations

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from ai.config import get_settings


def get_llm() -> ChatGoogleGenerativeAI:
    """Chat model configured from .env.

    No temperature is passed: gemini-3.6-flash uses fixed sampling
    defaults and emits a UserWarning on every call when it is supplied.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(model=settings.gemini_model)


def response_text(response: Any) -> str:
    """Get plain text out of a LangChain response.

    Newer Gemini models return `content` as a LIST of content blocks
    (e.g. [{"type": "text", "text": "SELECT ..."}]) rather than a string.
    Stringifying that list feeds Python's repr downstream instead of the
    actual output, so every block is unpacked explicitly here.
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


def ask(prompt: str) -> str:
    """One-shot prompt -> plain text."""
    return response_text(get_llm().invoke(prompt))