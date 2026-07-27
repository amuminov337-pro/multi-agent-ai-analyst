"""Probe the Gemini proxy before wiring it into the codebase.

This script does NOT touch ai/llm.py, ai/vectorstore.py, or any existing
client. It only checks, empirically, whether the proxy is reachable and
whether it actually returns Gemini-quality responses — the same "probe
before trusting" approach used for models (probe_models.py) and embedding
dimensions (F2).

    python scripts/probe_proxy.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("GEMINI_PROXY_API_KEY")
BASE_URL = os.getenv("GEMINI_PROXY_BASE_URL", "https://saidazam-litellm-proxy.hf.space")

CANDIDATE_MODELS = ["gemini-flash", "gemini-flash-lite", "gemini-embedding"]

PROBE_PROMPT = "Reply with exactly: PROXY OK"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_key_present() -> bool:
    section("1) KEY LOADED FROM .env")
    if not API_KEY:
        print("  [FAIL] GEMINI_PROXY_API_KEY not found in .env")
        return False
    masked = f"{API_KEY[:4]}...{API_KEY[-4:]}" if len(API_KEY) > 8 else "***"
    print(f"  [OK] key loaded ({masked}, len={len(API_KEY)})")
    print(f"  base url: {BASE_URL}")
    return True


def probe_chat_model(model_name: str) -> None:
    """Try one chat completion through the OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    print(f"\n  model: {model_name}")
    try:
        llm = ChatOpenAI(
            base_url=f"{BASE_URL}/v1",
            api_key=API_KEY,
            model=model_name,
        )
        response = llm.invoke(PROBE_PROMPT)
        text = getattr(response, "content", str(response))
        print(f"    [OK] response: {str(text)[:100]!r}")
    except Exception as exc:
        print(f"    [FAIL] {type(exc).__name__}: {str(exc)[:200]}")


def probe_embedding_model(model_name: str) -> None:
    """Try one embedding call and report the vector size."""
    from langchain_openai import OpenAIEmbeddings

    print(f"\n  model: {model_name}")
    try:
        emb = OpenAIEmbeddings(
            base_url=f"{BASE_URL}/v1",
            api_key=API_KEY,
            model=model_name,
        )
        vector = emb.embed_query("dimension probe")
        print(f"    [OK] embedding dimension: {len(vector)}")
    except Exception as exc:
        print(f"    [FAIL] {type(exc).__name__}: {str(exc)[:200]}")


def main() -> int:
    if not check_key_present():
        print("\nAdd GEMINI_PROXY_API_KEY to .env first.")
        return 1

    section("2) CHAT MODELS")
    for name in ("gemini-flash", "gemini-flash-lite"):
        probe_chat_model(name)

    section("3) EMBEDDING MODEL")
    probe_embedding_model("gemini-embedding")

    section("DONE")
    print("Review the responses above manually:")
    print("  - Did each model reply coherently (not an error, not gibberish)?")
    print("  - Does the embedding dimension look reasonable (usually 768-3072)?")
    print("This script changed nothing in the codebase — it only tested the proxy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())