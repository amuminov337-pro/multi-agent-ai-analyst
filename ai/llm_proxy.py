"""Alternate LLM access through the class proxy (temporary capacity relief).

This module exists ONLY because the primary Gemini quota (ai/llm.py, used by
F1-F10) was exhausted while finishing F11's RAGAS evaluation. It is a
SEPARATE entry point, not a replacement:

  * ai/llm.py (Google, direct)  -> used by F1-F10 and stays that way.
  * ai/llm_proxy.py (this file) -> used only where the primary quota is the
    blocker: the remaining F11 RAGAS run, and new F12-F14 code if needed.

The proxy is OpenAI-compatible, so it is wired through langchain-openai
rather than langchain-google-genai. Both are LangChain BaseChatModel
subclasses, which is what RAGAS's LangchainLLMWrapper actually requires — the
provider underneath does not matter to RAGAS.

Key is read from GEMINI_PROXY_API_KEY in .env, never hardcoded, never logged.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

_DEFAULT_BASE_URL = "https://saidazam-litellm-proxy.hf.space"


def _base_url() -> str:
    return os.getenv("GEMINI_PROXY_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _api_key() -> str:
    key = os.getenv("GEMINI_PROXY_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_PROXY_API_KEY is not set in .env — the proxy path cannot "
            "be used without it."
        )
    return key


@lru_cache(maxsize=1)
def get_proxy_llm(model: str = "gemini-flash-lite") -> ChatOpenAI:
    """Chat model through the proxy. Cached per process."""
    return ChatOpenAI(
        base_url=f"{_base_url()}/v1",
        api_key=_api_key(),
        model=model,
    )


@lru_cache(maxsize=1)
def get_proxy_embeddings(model: str = "gemini-embedding") -> OpenAIEmbeddings:
    """Embeddings through the proxy. Cached per process."""
    return OpenAIEmbeddings(
        base_url=f"{_base_url()}/v1",
        api_key=_api_key(),
        model=model,
    )


def ask_proxy(prompt: str, model: str = "gemini-flash-lite") -> str:
    """One-shot prompt -> plain text, through the proxy."""
    response = get_proxy_llm(model).invoke(prompt)
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)