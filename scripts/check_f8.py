"""F8 acceptance check — Critic / Verifier (7 pts).

Done-when (from the guide): "a deliberately wrong answer is caught and
revised; a good one is approved."

The critic is tested in ISOLATION: every answer/evidence pair below is
hand-built, so no answer-generation code is involved and a failure here can
only mean the critic itself is wrong.

Four parts:
  1. DETERMINISTIC VERDICTS (no LLM): an empty answer and an answer with no
     evidence behind it are rejected without a model call.
  2. LIVE JUDGEMENT: two sound answers must be approved; three deliberately
     broken ones (wrong number, invented fact, off-topic) must be rejected.
  3. REVISION CONTROL: route_after_critic() finishes on approval, revises on
     rejection, and stops revising once the cap is reached — the guarantee
     that the critic loop cannot run forever.
  4. STATE CONTRACT: the verdict keys are returned, the revision counter
     moves only on rejection, and reset_for_revision() reopens the agents
     while keeping the evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.agents.critic import (  # noqa: E402
    critic,
    precheck,
    reset_for_revision,
    route_after_critic,
)
from ai.config import get_settings  # noqa: E402
from ai.state import new_state  # noqa: E402

# --- reusable evidence fragments ----------------------------------------
SQL_CHURN = (
    "SQL: SELECT COUNT(*) FROM customers WHERE churned_on IS NOT NULL\n"
    "Result:\nCOUNT(*)\n3"
)
DOC_LEAVE = (
    "Every full-time employee receives 24 days of paid annual leave per "
    "calendar year, in addition to national public holidays. A maximum of 5 "
    "unused leave days may be carried over into the following year."
)
DOC_SLA = (
    "Severity 1 (production outage): first response within 1 hour, 24/7. "
    "Severity 2 (degraded functionality): first response within 4 business "
    "hours."
)


def build_state(
    question: str,
    answer: str,
    *,
    documents=None,
    sql_result=None,
    code_result=None,
    revisions: int = 0,
):
    """Hand-assemble a state so the critic is tested with nothing else running."""
    state = new_state(question)
    state["answer"] = answer
    state["documents"] = list(documents or [])
    state["sql_result"] = sql_result
    state["code_result"] = code_result
    state["revisions"] = revisions
    return state


# (label, state, expected verdict)
LIVE_CASES = [
    (
        "correct SQL-backed answer",
        build_state(
            "How many customers have churned?",
            "Three customers have churned.",
            sql_result=SQL_CHURN,
        ),
        True,
    ),
    (
        "correct document-backed answer",
        build_state(
            "How much annual leave do full-time employees get?",
            "Full-time employees receive 24 days of paid annual leave per year, "
            "on top of public holidays.",
            documents=[DOC_LEAVE],
        ),
        True,
    ),
    (
        "WRONG number (contradicts the SQL result)",
        build_state(
            "How many customers have churned?",
            "Seven customers have churned.",
            sql_result=SQL_CHURN,
        ),
        False,
    ),
    (
        "INVENTED fact (not in the evidence)",
        build_state(
            "How much annual leave do full-time employees get?",
            "Full-time employees receive 24 days of paid annual leave, plus a "
            "guaranteed annual bonus of 5,000 EUR and a company car.",
            documents=[DOC_LEAVE],
        ),
        False,
    ),
    (
        "OFF-TOPIC answer (evidence is unrelated)",
        build_state(
            "How fast must a severity 1 ticket get a first response?",
            "Employees get 24 days of annual leave.",
            documents=[DOC_SLA],
        ),
        False,
    ),
]


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def part_1_deterministic() -> bool:
    section("1) DETERMINISTIC VERDICTS — NO LLM CALL NEEDED")

    empty = build_state("How many customers have churned?", "", sql_result=SQL_CHURN)
    ungrounded = build_state(
        "How many customers have churned?", "Three customers have churned."
    )
    normal = build_state(
        "How many customers have churned?",
        "Three customers have churned.",
        sql_result=SQL_CHURN,
    )

    v_empty = precheck(empty)
    v_ungrounded = precheck(ungrounded)
    v_normal = precheck(normal)

    print(f"  empty answer      -> {v_empty}")
    print(f"  no evidence       -> {v_ungrounded}")
    print(f"  normal case       -> {v_normal} (None = hand over to the LLM)")

    checks = [
        ("empty answer rejected without an LLM", v_empty is not None and v_empty[0] is False),
        ("ungrounded answer rejected without an LLM", v_ungrounded is not None and v_ungrounded[0] is False),
        ("normal case deferred to the LLM", v_normal is None),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_2_live_judgement() -> bool:
    section("2) LIVE JUDGEMENT — GOOD APPROVED, BROKEN CAUGHT")
    failures = 0
    for label, state, expected in LIVE_CASES:
        update = critic(state)
        got = update["critic_ok"]
        ok = got is expected

        print(f"\n{label}")
        print(f"   question : {state['question']}")
        print(f"   answer   : {state['answer'][:90]}")
        print(f"   expected : {'approve' if expected else 'reject'}")
        print(f"   verdict  : {'approve' if got else 'reject'}")
        print(f"   reason   : {update['critic_reason'][:140]}")
        print(f"   revisions: {state['revisions']} -> {update['revisions']}")
        print(f"   result   : {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures += 1
    return failures == 0


def part_3_revision_control() -> bool:
    section("3) REVISION CONTROL — THE CRITIC LOOP CANNOT RUN FOREVER")
    settings = get_settings()
    cap = settings.max_revisions
    print(f"max_revisions = {cap}")

    approved = build_state("q", "a", sql_result=SQL_CHURN)
    approved["critic_ok"] = True
    approved["revisions"] = 0

    rejected_early = build_state("q", "a", sql_result=SQL_CHURN)
    rejected_early["critic_ok"] = False
    rejected_early["revisions"] = 1

    rejected_at_cap = build_state("q", "a", sql_result=SQL_CHURN)
    rejected_at_cap["critic_ok"] = False
    rejected_at_cap["revisions"] = cap

    rejected_over_cap = build_state("q", "a", sql_result=SQL_CHURN)
    rejected_over_cap["critic_ok"] = False
    rejected_over_cap["revisions"] = cap + 3

    cases = [
        ("approved -> finish", approved, "finish"),
        (f"rejected, {1}/{cap} used -> revise", rejected_early, "revise"),
        (f"rejected, cap reached ({cap}) -> finish", rejected_at_cap, "finish"),
        ("rejected, over cap -> finish", rejected_over_cap, "finish"),
    ]

    ok_all = True
    for name, state, expected in cases:
        route = route_after_critic(state)
        ok = route == expected
        ok_all = ok_all and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: got {route!r}")

    return ok_all


def part_4_state_contract() -> bool:
    section("4) STATE CONTRACT")

    good = build_state(
        "How many customers have churned?",
        "Three customers have churned.",
        sql_result=SQL_CHURN,
        revisions=0,
    )
    approved = critic(good)

    bad = build_state(
        "How many customers have churned?",
        "",  # empty -> deterministic rejection, no LLM call spent
        sql_result=SQL_CHURN,
        revisions=0,
    )
    rejected = critic(bad)

    reset_source = build_state(
        "How many customers have churned?",
        "wrong answer",
        sql_result=SQL_CHURN,
        revisions=1,
    )
    reset_source["visited"] = ["data", "code"]
    reset = reset_for_revision(reset_source)

    checks = [
        ("critic_ok returned", "critic_ok" in approved),
        ("critic_reason returned", "critic_reason" in approved),
        ("revisions returned", "revisions" in approved),
        ("steps returned", "steps" in approved),
        ("approval does NOT increment revisions", approved["revisions"] == 0),
        ("rejection increments revisions", rejected["revisions"] == 1),
        ("approved step is readable", "approved" in approved["steps"][-1]),
        ("rejected step is readable", "rejected" in rejected["steps"][-1]),
        ("original state untouched", good["critic_ok"] is False and good["revisions"] == 0),
        ("reset clears visited", reset.get("visited") == []),
        ("reset clears plan", reset.get("plan") == ""),
        ("reset keeps the evidence", reset_source["sql_result"] == SQL_CHURN),
        ("reset records the attempt", "revise" in reset["steps"][-1]),
    ]

    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def main() -> int:
    results = {
        "deterministic verdicts": part_1_deterministic(),
        "good approved, broken caught": part_2_live_judgement(),
        "revision loop bounded": part_3_revision_control(),
        "state contract": part_4_state_contract(),
    }

    section("RESULT")
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if all(results.values()):
        print("\nPASS — F8 done (7/7)")
        print("  - a deliberately wrong answer is caught and sent back")
        print("  - a correct, grounded answer is approved")
        print("  - invented facts are rejected even when they sound plausible")
        print("  - revisions are capped, so the critic loop always terminates")
        return 0
    print("\nFAIL — F8 not complete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())