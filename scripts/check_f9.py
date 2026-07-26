"""F9 acceptance check — Supervisor graph wiring (8 pts).

Done-when (from the guide): "a multi-part question runs end-to-end through
several agents and terminates with one answer."
Watch-out (rubric-mandated): "set a recursion/step limit so a mis-routing
loop can't run forever."

Five parts:
  1. STRUCTURE (no LLM): the graph compiles and every required node exists.
  2. RECURSION MATH (no LLM): the effective limit covers the true worst case
     derived from the agent count and the revision cap.
  3. MIS-ROUTING LOOP (no LLM): a stubbed supervisor that always picks the
     same agent is stopped by the limit instead of running forever.
  4. END-TO-END: a multi-part question reaches several agents and returns one
     verified answer containing both facts.
  5. CRITIC-OFF MODE: the same graph runs without the verification layer —
     the path F11 needs for its with/without-critic comparison.

It also writes documents/graph_diagram.mmd, which is Visual 1 of the final
submission.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.agents.supervisor import AGENT_ROUTES  # noqa: E402
from ai.config import get_settings  # noqa: E402
from ai.graph import (  # noqa: E402
    build_graph,
    get_graph,
    mermaid_diagram,
    required_recursion_limit,
    run,
    safe_recursion_limit,
)
from ai.state import new_state, push_step  # noqa: E402

DIAGRAM_PATH = ROOT / "documents" / "graph_diagram.mmd"

# Needs BOTH the database (a count) and the documents (a policy figure).
MULTI_PART_QUESTION = (
    "How many customers have churned, and what uptime does our contract "
    "commit us to?"
)
EXPECTED_FACTS = ["3", "99.9"]


def _fact_present(fact: str, answer: str) -> bool:
    """A fact matches as a standalone number OR as its English word form."""
    if re.search(rf"\b{re.escape(fact)}\b", answer):
        return True
    number_words = {"3": "three", "4": "four", "1": "one", "2": "two"}
    word = number_words.get(fact)
    if word and re.search(rf"\b{word}\b", answer, re.IGNORECASE):
        return True
    return False


EXPECTED_NODES = {"supervisor", "retriever", "web", "data", "code", "generate", "critic", "revise"}


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def part_1_structure() -> bool:
    section("1) GRAPH STRUCTURE")
    app = get_graph(use_critic=True)
    nodes = set(app.get_graph().nodes.keys())
    print(f"nodes compiled : {sorted(n for n in nodes if not n.startswith('__'))}")

    missing = EXPECTED_NODES - nodes
    checks = [
        ("graph compiles", app is not None),
        (f"all nodes present (missing: {sorted(missing) or 'none'})", not missing),
        ("critic-off variant compiles", build_graph(use_critic=False) is not None),
    ]

    DIAGRAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    diagram = mermaid_diagram()
    DIAGRAM_PATH.write_text(diagram, encoding="utf-8")
    print(f"\ndiagram written: {DIAGRAM_PATH}  ({len(diagram.splitlines())} lines)")
    print("  -> this file is Visual 1 of the final submission")
    checks.append(("mermaid diagram generated", "supervisor" in diagram))

    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_2_recursion_math() -> bool:
    section("2) RECURSION LIMIT COVERS THE WORST CASE")
    settings = get_settings()
    agents = len(AGENT_ROUTES)
    per_round = agents * 2 + 3
    rounds = settings.max_revisions + 1
    worst_case = rounds * per_round + settings.max_revisions

    computed = required_recursion_limit(settings.max_revisions)
    effective = safe_recursion_limit()

    print(f"agents                     : {agents}")
    print(f"max_revisions (.env)       : {settings.max_revisions}")
    print(f"steps per round            : {per_round}")
    print(f"rounds worst case          : {rounds}")
    print(f"worst-case steps           : {worst_case}")
    print(f"required limit (computed)  : {computed}")
    print(f"configured limit (.env)    : {settings.recursion_limit}")
    print(f"EFFECTIVE limit used       : {effective}")

    checks = [
        ("effective limit >= worst case", effective >= worst_case),
        ("effective limit >= computed minimum", effective >= computed),
        ("limit is finite and bounded", 0 < effective < 500),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if settings.recursion_limit < computed:
        print(
            f"  note: .env RECURSION_LIMIT={settings.recursion_limit} is below the "
            f"computed minimum {computed}; the code raised it automatically."
        )
    return all(ok for _, ok in checks)


def part_3_misrouting_loop() -> bool:
    section("3) MIS-ROUTING LOOP IS STOPPED BY THE LIMIT")
    print("Stub supervisor always picks 'data', bypassing F7's visited guard.")

    calls = {"n": 0}

    def stuck_supervisor(state):
        calls["n"] += 1
        return {"plan": "data", "steps": push_step(state, "supervisor→data [STUB]")}

    def fake_data(state):
        return {"sql_result": "stub", "steps": push_step(state, "data [STUB]")}

    app = build_graph(
        use_critic=True,
        node_overrides={"supervisor": stuck_supervisor, "data": fake_data},
    )

    limit = 12
    started = time.perf_counter()
    aborted = False
    try:
        app.invoke(new_state("loop probe"), config={"recursion_limit": limit})
    except Exception as exc:
        aborted = True
        print(f"stopped by      : {type(exc).__name__}")
    duration = time.perf_counter() - started

    print(f"supervisor calls: {calls['n']} (limit was {limit})")
    print(f"duration        : {duration:.2f}s")

    checks = [
        ("the loop was aborted, not run forever", aborted),
        ("aborted near the limit", calls["n"] <= limit + 2),
        ("aborted quickly", duration < 30),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_4_end_to_end() -> bool:
    section("4) END-TO-END — MULTI-PART QUESTION, ONE ANSWER")
    print(f"Q: {MULTI_PART_QUESTION}\n")

    started = time.perf_counter()
    final = run(MULTI_PART_QUESTION)
    duration = time.perf_counter() - started

    steps = final.get("steps") or []
    answer = (final.get("answer") or "").strip()
    visited = final.get("visited") or []

    print("trace:")
    for i, step in enumerate(steps, 1):
        print(f"  {i:>2}. {step}")

    print(f"\nagents used   : {visited}")
    print(f"revisions     : {final.get('revisions')}")
    print(f"critic_ok     : {final.get('critic_ok')}")
    print(f"critic_reason : {final.get('critic_reason')}")
    print(f"duration      : {duration:.1f}s")
    print(f"\nANSWER:\n{answer}\n")

    agent_hops = [s for s in steps if s.split("(")[0] in AGENT_ROUTES or s.startswith("data(")]
    facts_found = [f for f in EXPECTED_FACTS if _fact_present(f, answer)]

    checks = [
        ("run terminated with a result", bool(final)),
        ("exactly one answer produced", bool(answer)),
        ("several agents were used", len(set(visited)) >= 2),
        ("supervisor appears in the trace", any("supervisor" in s for s in steps)),
        ("generate appears in the trace", any("generate" in s for s in steps)),
        ("critic appears in the trace", any("critic" in s for s in steps)),
        (f"answer contains both facts {EXPECTED_FACTS} (found {facts_found})",
         len(facts_found) == len(EXPECTED_FACTS)),
        ("revisions within the cap", int(final.get("revisions", 0)) <= get_settings().max_revisions),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_5_critic_off() -> bool:
    section("5) CRITIC-OFF MODE (the path F11's comparison needs)")
    question = "How many customers have churned?"
    print(f"Q: {question}\n")

    final = run(question, use_critic=False)
    steps = final.get("steps") or []
    answer = (final.get("answer") or "").strip()

    print("trace:")
    for i, step in enumerate(steps, 1):
        print(f"  {i:>2}. {step}")
    print(f"\nANSWER:\n{answer}\n")

    checks = [
        ("run terminated", bool(final)),
        ("answer produced", bool(answer)),
        ("critic did NOT run", not any("critic" in s for s in steps)),
        ("no revisions happened", int(final.get("revisions", 0)) == 0),
        ("answer contains the correct count", _fact_present("3", answer)),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def main() -> int:
    results = {
        "graph structure": part_1_structure(),
        "recursion limit sound": part_2_recursion_math(),
        "mis-routing loop stopped": part_3_misrouting_loop(),
        "end-to-end multi-part answer": part_4_end_to_end(),
        "critic-off mode works": part_5_critic_off(),
    }

    section("RESULT")
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if all(results.values()):
        print("\nPASS — F9 done (8/8)")
        print("  - supervisor -> agents -> supervisor -> generate -> critic wired")
        print("  - a multi-part question runs through several agents to one answer")
        print("  - the step limit stops a mis-routing loop instead of hanging")
        print(f"  - Visual 1 saved to {DIAGRAM_PATH.relative_to(ROOT)}")
        return 0
    print("\nFAIL — F9 not complete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())