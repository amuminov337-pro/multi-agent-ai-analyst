"""Answer drafting node — turns collected evidence into the reply (F9).

This is the "generate" node the guide wires as the supervisor's "finish"
target. It is deliberately the ONLY place an answer is written, so the
critic (F8) always has exactly one artefact to judge.

Two behaviours matter for the graph as a whole:

  * On a REVISION it receives the critic's rejection reason and is told to
    fix that specific problem. Without this, a rejected answer would be
    redrafted blind and the revision loop would burn its budget repeating
    the same mistake.

  * With NO evidence it refuses deterministically instead of guessing. An
    invented answer would be rejected by the critic anyway, so spending a
    model call on it wastes quota and pollutes the trace.
"""

from __future__ import annotations

from ai.agents.critic import has_evidence
from ai.llm import ask
from ai.state import AgentState, evidence_bundle, push_step

NO_EVIDENCE_ANSWER = (
    "I could not answer this question: none of the specialist agents "
    "returned any usable evidence (no documents, no query result, no "
    "computation)."
)

GENERATE_PROMPT = """Answer the question using only the evidence below.

Question: {question}

Evidence gathered by the specialist agents:
{evidence}

Rules:
- Use only the evidence. Never add a fact, figure or name it does not contain.
- Copy exact numbers from SQL or code output; do not recompute them yourself.
- Be concise: two or three sentences unless the question needs more.
- If the evidence does not answer the question, say so plainly.
{revision_note}"""

REVISION_NOTE = """
IMPORTANT — a previous attempt was REJECTED by the verifier for this reason:
  {reason}
Write a new answer that fixes exactly that problem."""


def build_prompt(state: AgentState) -> str:
    """Assemble the drafting prompt, including the revision feedback if any."""
    reason = (state.get("critic_reason") or "").strip()
    revisions = int(state.get("revisions", 0))
    note = REVISION_NOTE.format(reason=reason) if (revisions and reason) else ""

    return GENERATE_PROMPT.format(
        question=state["question"],
        evidence=evidence_bundle(state),
        revision_note=note,
    )


def generate_agent(state: AgentState) -> dict:
    """LangGraph node: evidence -> state["answer"]. Never raises.

    A failed model call is recorded as a visible answer rather than an
    exception, so the critic still runs and the graph still terminates.
    """
    if not has_evidence(state):
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "steps": push_step(state, "generate(no evidence)"),
        }

    revisions = int(state.get("revisions", 0))
    try:
        answer = ask(build_prompt(state)).strip()
        label = f"generate(revision {revisions})" if revisions else "generate"
        if not answer:
            answer = NO_EVIDENCE_ANSWER
            label = "generate(empty output)"
    except Exception as exc:
        answer = f"Answer generation failed ({type(exc).__name__}): {exc}"
        label = f"generate(failed: {type(exc).__name__})"

    return {
        "answer": answer,
        "steps": push_step(state, label),
    }