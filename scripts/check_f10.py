"""F10 acceptance check — Long-term memory (5 pts).

Done-when (from the guide): "a follow-up like 'and the previous year?' uses
the earlier context correctly."

Five parts:
  1. STORE AND RECALL: turns are written to the memory collection and the
     relevant one comes back for a related question.
  2. POISON GUARD (no LLM): should_remember() refuses empty, failed and
     critic-rejected answers, so a wrong answer can never re-enter the
     evidence bundle on a later turn.
  3. FOLLOW-UP END-TO-END: turn 1 establishes context, turn 2 is a fragment
     that is meaningless on its own, and the system still answers it correctly.
  4. MEMORY OFF: use_memory=False neither recalls nor writes — the
     reproducible mode F11's evaluation needs.
  5. RESILIENCE: a broken memory store degrades to a no-op instead of
     breaking the run.

The memory collection is reset at the start so the run is deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ai.memory as memory_module  # noqa: E402
from ai.config import get_settings  # noqa: E402
from ai.graph import run  # noqa: E402
from ai.memory import (  # noqa: E402
    memory_size,
    recall,
    remember,
    reset_memory,
    should_remember,
)

# Turn 1 establishes "employee headcount by department"; turn 2 is a fragment.
TURN_1 = "How many employees work in the Engineering department?"
TURN_1_TRUTH = ("4", "four")
TURN_2 = "What about Data Science?"
TURN_2_TRUTH = ("3", "three")


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def has_fact(answer: str, forms) -> bool:
    """Match a number as a digit or as its English word."""
    low = answer.lower()
    return any(form.lower() in low for form in forms)


def part_1_store_and_recall() -> bool:
    section("1) STORE AND RECALL")
    settings = get_settings()
    print(f"memory collection : {settings.memory_collection}")
    print(f"reset             : {reset_memory()}")
    print(f"size after reset  : {memory_size()}")

    wrote = [
        remember("How many customers have churned?", "Three customers have churned."),
        remember(
            "What is the hotel expense limit in European cities?",
            "180 EUR per night in European cities.",
        ),
        remember(
            "How fast must a severity 1 ticket be answered?",
            "Within 1 hour, 24/7.",
        ),
    ]
    size = memory_size()
    print(f"turns written     : {wrote}")
    print(f"size after writes : {size}")

    hits = recall("Tell me about the hotel spending limit")
    print(f"\nrecall query      : 'Tell me about the hotel spending limit'")
    print(f"turns recalled    : {len(hits)}")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] {hit.strip().replace(chr(10), ' | ')[:100]}")

    blob = " ".join(hits)
    checks = [
        ("all three turns written", all(wrote)),
        ("collection holds three turns", size == 3),
        ("recall returned something", bool(hits)),
        ("recall found the relevant turn (180 EUR)", "180" in blob),
        ("empty question is not stored", remember("", "x") is False),
        ("empty answer is not stored", remember("x", "") is False),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_2_poison_guard() -> bool:
    section("2) POISON GUARD — WRONG ANSWERS NEVER ENTER MEMORY")

    approved = {"answer": "Three customers have churned.", "critic_ok": True}
    rejected = {"answer": "Seven customers have churned.", "critic_ok": False}
    empty = {"answer": "", "critic_ok": True}
    failed = {
        "answer": "Answer generation failed (RuntimeError): boom",
        "critic_ok": True,
    }
    aborted = {
        "answer": "The run was stopped by the step limit before an answer was verified.",
        "critic_ok": False,
    }
    no_critic = {"answer": "Three customers have churned.", "critic_ok": False}

    cases = [
        ("critic approved -> store", should_remember(approved, True), True),
        ("critic rejected -> DO NOT store", should_remember(rejected, True), False),
        ("empty answer -> DO NOT store", should_remember(empty, True), False),
        ("generation failed -> DO NOT store", should_remember(failed, True), False),
        ("run aborted -> DO NOT store", should_remember(aborted, True), False),
        ("critic disabled -> store anyway", should_remember(no_critic, False), True),
    ]

    ok_all = True
    for name, got, expected in cases:
        ok = got is expected
        ok_all = ok_all and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: got {got}")
    return ok_all


def part_3_followup() -> bool:
    section("3) FOLLOW-UP — THE DONE-WHEN CONDITION")
    reset_memory()

    print(f"TURN 1: {TURN_1}")
    first = run(TURN_1)
    answer_1 = (first.get("answer") or "").strip()
    print(f"  answer   : {answer_1}")
    print(f"  critic_ok: {first.get('critic_ok')}")
    print(f"  steps    : {first.get('steps')}")
    stored = memory_size()
    print(f"  memory   : {stored} turn(s) stored")

    print(f"\nTURN 2 (a fragment, meaningless alone): {TURN_2}")
    second = run(TURN_2)
    answer_2 = (second.get("answer") or "").strip()
    steps_2 = second.get("steps") or []
    print(f"  answer   : {answer_2}")
    print(f"  critic_ok: {second.get('critic_ok')}")
    print("  trace    :")
    for i, step in enumerate(steps_2, 1):
        print(f"    {i:>2}. {step}")
    print(f"  memory recalled into state: {len(second.get('memory') or [])} turn(s)")

    condensed = [s for s in steps_2 if "condensed" in s]
    checks = [
        ("turn 1 answered correctly (4)", has_fact(answer_1, TURN_1_TRUTH)),
        ("turn 1 was stored in memory", stored >= 1),
        ("turn 2 recalled earlier context", bool(second.get("memory"))),
        (f"turn 2 was condensed ({condensed[0] if condensed else 'no'})", bool(condensed)),
        ("turn 2 answered correctly (3)", has_fact(answer_2, TURN_2_TRUTH)),
        ("turn 2 was verified by the critic", bool(second.get("critic_ok"))),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_4_memory_off() -> bool:
    section("4) MEMORY OFF — REPRODUCIBLE MODE FOR F11")
    reset_memory()
    remember("Seeded turn", "This must not be recalled.")
    before = memory_size()

    final = run("How many customers have churned?", use_memory=False)
    after = memory_size()
    steps = final.get("steps") or []

    print(f"memory before : {before}")
    print(f"memory after  : {after}")
    print(f"state memory  : {final.get('memory')}")
    print(f"answer        : {(final.get('answer') or '').strip()}")

    checks = [
        ("nothing was recalled", not (final.get("memory") or [])),
        ("nothing was condensed", not any("condensed" in s for s in steps)),
        ("nothing was written", after == before),
        ("the run still worked", has_fact(final.get("answer") or "", ("3", "three"))),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_5_resilience() -> bool:
    section("5) RESILIENCE — A BROKEN STORE IS A NO-OP, NOT A CRASH")
    original_store = memory_module._store
    original_count = memory_module.collection_count

    def broken_store(recreate: bool = False):
        raise RuntimeError("simulated Qdrant outage")

    def broken_count(collection=None):
        raise RuntimeError("simulated Qdrant outage")

    memory_module._store = broken_store
    memory_module.collection_count = broken_count
    try:
        recalled = recall("anything")
        wrote = remember("q", "a")
        size = memory_size()
        print(f"recall with broken store  : {recalled}")
        print(f"remember with broken store: {wrote}")
        print(f"size with broken store    : {size}")

        checks = [
            ("recall returned empty, no exception", recalled == []),
            ("remember reported failure, no exception", wrote is False),
            ("size reported 0, no exception", size == 0),
        ]
    finally:
        memory_module._store = original_store
        memory_module.collection_count = original_count

    print(f"store restored, size now  : {memory_size()}")
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def main() -> int:
    results = {
        "store and recall": part_1_store_and_recall(),
        "poison guard": part_2_poison_guard(),
        "follow-up uses earlier context": part_3_followup(),
        "memory can be disabled": part_4_memory_off(),
        "broken store degrades safely": part_5_resilience(),
    }

    section("RESULT")
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if all(results.values()):
        print("\nPASS — F10 done (5/5)")
        print("  - past turns are stored in a separate memory collection")
        print("  - a follow-up fragment is resolved against earlier context")
        print("  - only critic-approved answers are remembered")
        print("  - memory can be switched off for reproducible evaluation")
        return 0
    print("\nFAIL — F10 not complete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())