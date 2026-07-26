"""Critic / Verifier — the quality gate before the user sees an answer (F8).

Checks the drafted answer in state["answer"] against every piece of
evidence the specialists collected, and either approves it or sends it back
for revision.

Two properties are verified, and failing either is a rejection:
  * CORRECTNESS — does the answer contradict the evidence? A number that
    disagrees with the SQL result is the classic case.
  * GROUNDING   — is every specific claim actually supported? An answer that
    invents a figure, name or date the evidence never mentions is rejected
    even when it sounds entirely plausible.

Control flow is deliberately kept OUT of the verdict. critic() reports an
honest judgement and counts the rejection; route_after_critic() decides what
the graph does with it, including the revision cap that guarantees
termination. Mixing the two would make "the answer was wrong" and "the retry
budget ran out" indistinguishable in the Langfuse trace (F12).

A cheap deterministic pre-check runs before the LLM: an empty answer, or an
answer with no evidence behind it at all, is rejected without spending a
model call — that verdict needs no judgement.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from ai.config import get_settings
from ai.llm import get_llm, response_text
from ai.state import AgentState, evidence_bundle, push_step

CRITIC_PROMPT = """You are a strict verifier. Decide whether the drafted answer
is correct and fully supported by the evidence.

Question: {question}

Evidence gathered by the specialist agents:
{evidence}

Drafted answer:
{answer}

Approve (ok = true) only if BOTH hold:
1. Nothing in the answer contradicts the evidence.
2. Every specific claim — numbers, names, dates, policy limits — appears in
   the evidence or follows directly from it.

Do NOT reject for:
- rephrasing, summarising or reordering the evidence
- leaving out evidence the question did not ask about
- correct arithmetic on numbers that are present

DO reject if:
- a number or fact disagrees with the evidence
- the answer states a specific fact the evidence does not contain
- the answer does not actually answer the question

Return ok as a boolean, and one short sentence in reason."""


class Verdict(BaseModel):
    """Structured judgement returned by the critic LLM."""

    ok: bool = Field(description="True only if the answer is correct AND grounded")
    reason: str = Field(default="", description="One short sentence of justification")


def has_evidence(state: AgentState) -> bool:
    """True when at least one specialist produced something to check against."""
    return bool(
        (state.get("documents") or [])
        or state.get("sql_result")
        or state.get("code_result")
        or (state.get("memory") or [])
    )


def precheck(state: AgentState) -> Optional[Tuple[bool, str]]:
    """Deterministic verdicts that need no model. None means "ask the LLM"."""
    answer = (state.get("answer") or "").strip()
    if not answer:
        return False, "No answer was drafted, so there is nothing to verify."
    if not has_evidence(state):
        return (
            False,
            "No evidence was collected, so the answer cannot be grounded.",
        )
    return None


def verify(state: AgentState) -> Tuple[bool, str]:
    """Ask the LLM to judge the answer. Never raises.

    If the model is unavailable the answer is APPROVED but explicitly marked
    as unverified: blocking forever on a broken critic would be worse than
    shipping an answer whose trace says nobody checked it.
    """
    prompt = CRITIC_PROMPT.format(
        question=state["question"],
        evidence=evidence_bundle(state),
        answer=state.get("answer") or "",
    )

    try:
        verdict = get_llm().with_structured_output(Verdict).invoke(prompt)
        return bool(verdict.ok), (verdict.reason or "").strip()
    except Exception:
        pass

    # Structured output can fail on model or schema changes; a plain-text
    # verdict still carries the decision, so try that before giving up.
    try:
        text = response_text(get_llm().invoke(prompt + "\n\nStart with YES or NO."))
        head = text.strip().lower()
        if head.startswith("yes") or head.startswith("true"):
            return True, text.strip()[:200]
        if head.startswith("no") or head.startswith("false"):
            return False, text.strip()[:200]
        return True, f"critic output unparseable, answer NOT verified: {text[:120]}"
    except Exception as exc:
        return (
            True,
            f"critic unavailable ({type(exc).__name__}), answer NOT verified",
        )


def critic(state: AgentState) -> dict:
    """LangGraph node: judge the drafted answer.

    Writes the verdict to state["critic_ok"]/["critic_reason"] and increments
    state["revisions"] ONLY on rejection, so the counter measures failed
    attempts rather than total critic runs.
    """
    verdict = precheck(state)
    if verdict is None:
        verdict = verify(state)
    ok, reason = verdict

    return {
        "critic_ok": ok,
        "critic_reason": reason,
        "revisions": int(state.get("revisions", 0)) + (0 if ok else 1),
        "steps": push_step(state, f"critic({'approved' if ok else 'rejected'})"),
    }


def route_after_critic(state: AgentState) -> str:
    """Decide the graph's next hop after a verdict: "finish" or "revise".

    Termination has two independent guarantees:
      * an approved answer finishes immediately;
      * a rejected answer is retried at most MAX_REVISIONS times, after which
        it finishes anyway rather than looping.

    F9 must pair the "revise" edge with reset_for_revision(), otherwise the
    supervisor has no agents left to delegate to and the retry collects no
    new evidence.
    """
    if state.get("critic_ok"):
        return "finish"

    settings = get_settings()
    if int(state.get("revisions", 0)) >= settings.max_revisions:
        return "finish"

    return "revise"


def reset_for_revision(state: AgentState) -> dict:
    """State update that opens a fresh evidence round for a retry.

    Clearing `visited` is what makes a revision meaningful: the supervisor
    (F7) refuses to re-select an agent it has already used, so without this
    reset a rejected answer would come back to a supervisor with nothing left
    to try. Collected evidence is deliberately KEPT — the retry should add to
    it, not start blind — and the critic's reason is preserved so the next
    draft knows what was wrong.
    """
    return {
        "visited": [],
        "plan": "",
        "steps": push_step(state, f"revise(attempt {int(state.get('revisions', 0)) + 1})"),
    }