"""Probe which Gemini models this API key can ACTUALLY call.

Listing a model via models.list() does not mean the account may call it:
a model can be listed and still return 404 "no longer available to new
users". The only reliable test is a real call, so this script sends the
smallest possible prompt to each candidate and reports what happens.

Quota is counted per project PER MODEL, so probing many models costs one
request from each model's own bucket — it cannot exhaust a single model.

    python scripts/probe_models.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from ai.config import get_settings  # noqa: E402

PROMPT = "Reply with the two characters: OK"

# Substrings that mark a model as irrelevant for chat/tool use.
SKIP_MARKERS = (
    "embedding",
    "aqa",
    "tts",
    "image",
    "imagen",
    "veo",
    "vision",
    "learnlm",
    "gemma",
    "live",
    "native-audio",
    "thinking-exp",
)


def list_candidates() -> list:
    """Every generateContent model, minus the obviously irrelevant ones."""
    import google.generativeai as genai

    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)

    names = []
    for model in genai.list_models():
        if "generateContent" not in getattr(model, "supported_generation_methods", []):
            continue
        name = model.name.replace("models/", "")
        if any(marker in name.lower() for marker in SKIP_MARKERS):
            continue
        names.append(name)
    return sorted(set(names))


def probe(name: str) -> tuple:
    """Try one tiny call. Returns (status, detail)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        response = ChatGoogleGenerativeAI(model=name).invoke(PROMPT)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            text = str(content)
        return "WORKS", text.strip()[:40]
    except Exception as exc:
        message = str(exc)
        if "404" in message or "no longer available" in message:
            return "DEPRECATED", "not available to this account"
        if "429" in message or "quota" in message.lower():
            return "QUOTA", "callable, but daily quota already spent"
        if "403" in message or "permission" in message.lower():
            return "NO ACCESS", "permission denied"
        return "ERROR", f"{type(exc).__name__}: {message[:80]}"


def main() -> int:
    candidates = list_candidates()
    print(f"Probing {len(candidates)} candidate models with a 1-token prompt.\n")

    results = {}
    for name in candidates:
        status, detail = probe(name)
        results.setdefault(status, []).append(name)
        print(f"  [{status:<10}] {name:<42} {detail}")

    print("\n" + "=" * 60)
    print("USABLE MODELS (pick one for GEMINI_MODEL in .env)")
    print("=" * 60)
    usable = results.get("WORKS", [])
    quota = results.get("QUOTA", [])

    if usable:
        for name in usable:
            print(f"  {name}")
    else:
        print("  (none responded successfully)")

    if quota:
        print("\nCallable but out of quota today (usable after the reset):")
        for name in quota:
            print(f"  {name}")

    print(f"\ndeprecated / no access: {len(results.get('DEPRECATED', []))} model(s)")
    return 0 if (usable or quota) else 1


if __name__ == "__main__":
    raise SystemExit(main())