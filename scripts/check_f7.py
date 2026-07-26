"""F7 acceptance check — Supervisor / Router (10 pts).

Done-when (from the guide): "for a SQL question it routes to `data`; for a
doc question to `retriever`; it eventually chooses `finish`."

Four parts:
  1. DETERMINISTIC GUARD (no LLM): enforce_route() is tested directly with
     every abusive input — a repeated agent, garbage, an empty string, a
     fully-visited state. This is the part that guarantees termination, so
     it is tested without any model in the loop.
  2. LIVE ROUTING: four question types, each must reach the right agent.
  3. TERMINATION: a state whose evidence is complete, and a state where
     every agent has run, must both yield "finish".
  4. STATE CONTRACT: the node returns plan/visited/steps and visited grows
     by exactly one per delegation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.agents.supervisor import (  # noqa: E402
    AGENT_ROUTES,
    available_agents,
    enforce_route,
    supervisor,
)
from ai.state import new_state  # noqa: E402

# (question, expected first route, why)
ROUTING_CASES = [
    (
        "How many customers have churned?",
        "data",
        "a count over company records -> SQL",
    ),
    (
        "What is our annual leave policy for full-time employees?",
        "retriever",
        "an internal policy -> documents",
    ),
    (
        "What is 2 to the power of 40 divided by 7, rounded to 2 decimals?",
        "code",
        "pure arithmetic -> Python",
    ),
    (
        "What are the latest EU regulations on AI announced this year?",
        "web",
        "public current information -> web search",
    ),
]


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def part_1_guard() -> bool:
    section("1) DETERMINISTIC GUARD — enforce_route() WITHOUT AN LLM")

    fresh = new_state("dummy question")

    used_data = new_state("dummy question")
    used_data["visited"] = ["data"]

    all_used = new_state("dummy question")
    all_used["visited"] = list(AGENT_ROUTES)

    cases = [
        ("clean choice accepted", "data", fresh, "data"),
        ("whitespace tolerated", "  retriever \n", fresh, "retriever"),
        ("case tolerated", "DATA", fresh, "data"),
        ("arrow form tolerated", "supervisor -> code", fresh, "code"),
        ("quoted form tolerated", '"web"', fresh, "web"),
        ("finish accepted", "finish", used_data, "finish"),
        ("repeat agent blocked", "data", used_data, "finish"),
        ("unknown label, mid-run", "banana", used_data, "finish"),
        ("unknown label, first hop", "banana", fresh, "retriever"),
        ("empty label, first hop", "", fresh, "retriever"),
        ("none, first hop", None, fresh, "retriever"),
        ("all agents used", "code", all_used, "finish"),
        ("all agents used, finish", "finish", all_used, "finish"),
    ]

    ok_all = True
    for name, raw, state, expected in cases:
        route, note = enforce_route(raw, state)
        ok = route == expected
        ok_all = ok_all and ok
        detail = f" [{note}]" if note else ""
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {raw!r} -> {route}{detail}")
        if not ok:
            print(f"         expected {expected!r}")

    leftover = available_agents(used_data)
    ok = "data" not in leftover
    ok_all = ok_all and ok
    print(f"  [{'OK' if ok else 'FAIL'}] visited agent removed from available: {leftover}")

    return ok_all


def part_2_live_routing() -> bool:
    section("2) LIVE ROUTING — RIGHT AGENT PER QUESTION TYPE")
    failures = 0
    for question, expected, why in ROUTING_CASES:
        state = new_state(question)
        update = supervisor(state)
        route = update["plan"]
        ok = route == expected
        print(f"\nQ: {question}")
        print(f"   expected : {expected}  ({why})")
        print(f"   routed to: {route}")
        print(f"   step     : {update['steps'][-1]}")
        print(f"   result   : {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures += 1
    return failures == 0


def part_3_termination() -> bool:
    section("3) TERMINATION — THE SUPERVISOR EVENTUALLY FINISHES")

    satisfied = new_state("How many customers have churned?")
    satisfied["visited"] = ["data"]
    satisfied["sql_result"] = (
        "SQL: SELECT COUNT(*) FROM customers WHERE churned_on IS NOT NULL\n"
        "Result:\nCOUNT(*)\n3"
    )
    update_a = supervisor(satisfied)
    ok_a = update_a["plan"] == "finish"
    print(f"  [{'OK' if ok_a else 'FAIL'}] complete evidence -> {update_a['plan']}")
    print(f"         step: {update_a['steps'][-1]}")

    exhausted = new_state("Some very open-ended question")
    exhausted["visited"] = list(AGENT_ROUTES)
    update_b = supervisor(exhausted)
    ok_b = update_b["plan"] == "finish"
    print(f"  [{'OK' if ok_b else 'FAIL'}] all agents used -> {update_b['plan']}")
    print(f"         step: {update_b['steps'][-1]}")

    state = new_state("How many employees are there and what is the leave policy?")
    hops = []
    for _ in range(10):
        update = supervisor(state)
        state = {**state, **update}
        if update["plan"] == "finish":
            break
        hops.append(update["plan"])
    ok_c = len(hops) <= len(AGENT_ROUTES) and state["plan"] == "finish"
    print(f"  [{'OK' if ok_c else 'FAIL'}] bounded delegations: {hops} then finish")
    print(f"         at most {len(AGENT_ROUTES)} hops possible, got {len(hops)}")

    return ok_a and ok_b and ok_c


def part_4_state_contract() -> bool:
    section("4) STATE CONTRACT")
    state = new_state("How many customers have churned?")
    update = supervisor(state)

    checks = [
        ("plan key returned", "plan" in update),
        ("steps key returned", "steps" in update),
        ("visited key returned on delegation", "visited" in update),
        ("visited grew by one", len(update.get("visited", [])) == 1),
        ("visited holds the chosen agent", update.get("visited") == [update["plan"]]),
        ("step label readable", update["steps"][-1].startswith("supervisor→")),
        ("original state untouched", state["visited"] == [] and state["plan"] == ""),
    ]

    done = new_state("dummy")
    done["visited"] = list(AGENT_ROUTES)
    finish_update = supervisor(done)
    checks.append(("finish does not mark a visit", "visited" not in finish_update))

    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def main() -> int:
    results = {
        "deterministic guard": part_1_guard(),
        "routes to the right agent": part_2_live_routing(),
        "always terminates": part_3_termination(),
        "state contract": part_4_state_contract(),
    }

    section("RESULT")
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if all(results.values()):
        print("\nPASS — F7 done (10/10)")
        print("  - SQL question routes to data, doc question to retriever")
        print("  - math routes to code, current events to web")
        print("  - repeated or invalid routes are blocked in code, not by prompt")
        print("  - the supervisor always reaches finish within four hops")
        return 0
    print("\nFAIL — F7 not complete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())