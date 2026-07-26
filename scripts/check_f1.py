"""F1 acceptance check.

Done when:
  1. every key loads from .env (required present, optional flagged);
  2. AgentState is defined and actually usable by nodes.

Run from the capstone/ directory:
    python scripts/check_f1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.config import ConfigError, ENV_PATH, get_settings  # noqa: E402
from ai.state import (  # noqa: E402
    STATE_KEYS,
    AgentState,
    evidence_bundle,
    new_state,
    push_step,
)

EXPECTED_KEYS = {
    "question",
    "plan",
    "visited",
    "documents",
    "sql_result",
    "code_result",
    "memory",
    "answer",
    "critic_ok",
    "critic_reason",
    "steps",
    "revisions",
}

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def mask(value: str | None) -> str:
    if not value:
        return "(not set)"
    return f"{value[:6]}...{value[-4:]} (len={len(value)})"


def check_env() -> None:
    print("\n1) CONFIG / .env")
    if not ENV_PATH.exists():
        fail(f".env not found at {ENV_PATH}")
        return

    try:
        s = get_settings()
    except ConfigError as exc:
        fail(str(exc))
        return

    ok(f"GOOGLE_API_KEY loaded  -> {mask(s.google_api_key)}")

    if s.qdrant_mode == "cloud":
        ok(f"QDRANT_URL loaded      -> {s.qdrant_url}")
        ok(f"QDRANT_API_KEY loaded  -> {mask(s.qdrant_api_key)}")
    else:
        ok(f"Qdrant embedded mode   -> {s.qdrant_path}")

    if s.tavily_enabled:
        ok(f"TAVILY_API_KEY loaded  -> {mask(s.tavily_api_key)}")
    else:
        print("  [WARN] TAVILY_API_KEY not set -> F4 web agent will skip")

    if s.langfuse_enabled:
        ok(f"LANGFUSE keys loaded   -> {mask(s.langfuse_public_key)}")
        ok(f"LANGFUSE_HOST          -> {s.langfuse_host}")
    else:
        print("  [WARN] LANGFUSE keys not set -> F12 tracing disabled")

    if s.max_revisions <= 0:
        fail("MAX_REVISIONS must be >= 1 so the critic loop can terminate")
    else:
        ok(f"graph limits           -> max_revisions={s.max_revisions}, "
           f"recursion_limit={s.recursion_limit}")

    print("\n  --- settings summary (no secrets) ---")
    for line in s.describe().splitlines():
        print(f"  {line}")


def check_state() -> None:
    print("\n2) SHARED STATE")

    if set(STATE_KEYS) != EXPECTED_KEYS:
        missing = EXPECTED_KEYS - set(STATE_KEYS)
        extra = set(STATE_KEYS) - EXPECTED_KEYS
        fail(f"AgentState key mismatch (missing={missing}, extra={extra})")
        return
    ok(f"AgentState defines all {len(EXPECTED_KEYS)} keys")

    state: AgentState = new_state("How many customers churned last quarter?")
    if set(state.keys()) != EXPECTED_KEYS:
        fail("new_state() did not initialise every key")
        return
    ok("new_state() returns a fully-initialised state (no KeyError possible)")

    if state["revisions"] != 0 or state["steps"] != []:
        fail("new_state() must start with revisions=0 and steps=[]")
    else:
        ok("termination counters start clean (revisions=0, steps=[])")

    update_a = {
        "documents": state["documents"] + ["churn policy doc chunk"],
        "steps": push_step(state, "retriever"),
    }
    state = {**state, **update_a}

    update_b = {
        "sql_result": "SELECT COUNT(*) ... -> 42",
        "steps": push_step(state, "data(sql)"),
    }
    state = {**state, **update_b}

    if state["steps"] != ["retriever", "data(sql)"]:
        fail(f"push_step() lost the trace: {state['steps']}")
    else:
        ok("push_step() accumulates the agent trace: retriever -> data(sql)")

    if len(state["documents"]) != 1 or state["sql_result"] is None:
        fail("partial node updates did not merge correctly")
    else:
        ok("partial node updates merge without clobbering earlier evidence")

    bundle = evidence_bundle(state)
    if "[doc]" not in bundle or "[sql]" not in bundle:
        fail("evidence_bundle() did not collect all evidence for the critic")
    else:
        ok("evidence_bundle() feeds the critic with every collected source")

    try:
        new_state("   ")
        fail("new_state() accepted an empty question")
    except ValueError:
        ok("new_state() rejects an empty question")


def check_gemini_live() -> None:
    print("\n3) GEMINI LIVE CALL (proves the key actually works)")
    try:
        import google.generativeai as genai
    except ImportError:
        fail("google-generativeai not installed (pip install google-generativeai)")
        return

    try:
        s = get_settings()
    except ConfigError:
        fail("cannot run live check: config failed above")
        return

    genai.configure(api_key=s.google_api_key)

    try:
        usable = [
            m.name
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
    except Exception as exc:  # noqa: BLE001
        fail(f"could not list models: {type(exc).__name__}: {exc}")
        return

    ok(f"API key valid — {len(usable)} models support generateContent")
    print("  sample models:")
    for name in usable[:8]:
        print(f"    - {name}")

    configured = s.gemini_model
    matches = [n for n in usable if configured in n]
    if matches:
        ok(f"configured GEMINI_MODEL '{configured}' is available")
    else:
        fail(
            f"GEMINI_MODEL '{configured}' not in the available list — "
            f"pick one from above and set it in .env"
        )
        return

    try:
        model = genai.GenerativeModel(matches[0])
        reply = model.generate_content("Reply with exactly: F1 OK")
        text = (reply.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        fail(f"generate_content failed: {type(exc).__name__}: {exc}")
        return

    ok(f"live generation succeeded -> {text!r}")


def main() -> int:
    print("=" * 62)
    print("F1 CHECK — shared state & config")
    print("=" * 62)

    check_env()
    check_state()
    check_gemini_live()

    print("\n" + "=" * 62)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} problem(s))")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("RESULT: PASS — F1 done (5/5)")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())