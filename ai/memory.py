"""Long-term memory over past turns (F10).

Each completed question/answer pair is stored in its OWN Qdrant collection
(MEMORY_COLLECTION, kept separate from the document corpus so a remembered
answer can never be mistaken for a source document) and the relevant ones are
recalled on the next turn.

Two decisions carry most of the weight here:

  * ONLY VERIFIED ANSWERS ARE STORED. A rejected answer written to memory
    would come back on a later turn as [memory] evidence, and the generator
    would treat it as established fact — one wrong answer would keep
    re-entering the evidence bundle and compound. The critic (F8) is the gate;
    should_remember() enforces it.

  * A FOLLOW-UP IS CONDENSED BEFORE ROUTING. "What about Data Science?" is
    meaningless to the SQL agent, which only ever sees state["question"].
    Feeding memory to the supervisor alone makes the ROUTING right while
    leaving the generated SQL nonsense. condense_question() rewrites the
    fragment into a standalone question, so every downstream agent works
    unchanged.

Memory is an enhancement, never a dependency: every function here degrades to
a no-op on failure, so a Qdrant outage cannot break a run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from ai.agents.generate import NO_EVIDENCE_ANSWER
from ai.config import get_settings
from ai.llm import ask
from ai.vectorstore import collection_count, get_vectorstore

#: Answer prefixes that mark a failed run — never worth remembering.
FAILURE_MARKERS = (
    "Answer generation failed",
    "The run was stopped by the step limit",
    NO_EVIDENCE_ANSWER[:40],
)

CONDENSE_PROMPT = """Rewrite the user's question so it stands on its own.

Earlier turns in this conversation:
{history}

New question: {question}

Rules:
- If the new question already makes sense on its own, return it UNCHANGED.
- If it depends on the earlier turns (for example "what about X?", "and the
  previous year?", "how about theirs?"), rewrite it as a full question that
  carries the missing subject over from those turns.
- Keep the user's intent exactly. Never answer the question.
- Output only the rewritten question, nothing else."""


def _store(recreate: bool = False):
    """Vector store bound to the memory collection."""
    settings = get_settings()
    return get_vectorstore(collection=settings.memory_collection, recreate=recreate)


def format_turn(question: str, answer: str) -> str:
    """One memory record. Timestamped so recall order is auditable."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"[{stamp}]\nQ: {question.strip()}\nA: {answer.strip()}"


def memory_size() -> int:
    """Number of stored turns. Returns 0 if the store is unreachable."""
    settings = get_settings()
    try:
        return collection_count(settings.memory_collection)
    except Exception:
        return 0


def reset_memory() -> bool:
    """Drop every stored turn. Used by the F10 check for a deterministic start."""
    try:
        _store(recreate=True)
        return True
    except Exception:
        return False


def should_remember(state: dict, critic_enabled: bool = True) -> bool:
    """Decide whether this turn is safe to store. Pure and testable.

    Rejects: an empty answer, a run that failed or aborted, and — when the
    critic was active — anything the critic did not approve.
    """
    answer = (state.get("answer") or "").strip()
    if not answer:
        return False
    if any(answer.startswith(marker) for marker in FAILURE_MARKERS):
        return False
    if critic_enabled and not state.get("critic_ok"):
        return False
    return True


def remember(question: str, answer: str) -> bool:
    """Store one turn. Returns whether it was written; never raises."""
    if not (question or "").strip() or not (answer or "").strip():
        return False
    try:
        _store().add_texts([format_turn(question, answer)])
        return True
    except Exception:
        return False


def recall(question: str, k: Optional[int] = None) -> List[str]:
    """Most relevant past turns for this question. Never raises.

    An empty list is returned both when nothing is relevant and when the
    store is unreachable — callers treat the two identically.
    """
    settings = get_settings()
    top_k = k if k is not None else settings.memory_k
    try:
        if memory_size() == 0:
            return []
        hits = _store().similarity_search(question, k=top_k)
        return [hit.page_content for hit in hits]
    except Exception:
        return []


def condense_question(question: str, memory: List[str]) -> str:
    """Rewrite a follow-up into a standalone question using past turns.

    Returns the question unchanged when there is no memory, when the model
    declines to rewrite it, or on any failure — the pipeline must never be
    blocked by this step.
    """
    if not memory:
        return question

    history = "\n\n".join(memory)
    try:
        rewritten = ask(
            CONDENSE_PROMPT.format(history=history, question=question)
        ).strip()
    except Exception:
        return question

    # Guard against a model that answers instead of rewriting, or returns junk.
    if not rewritten or len(rewritten) > 400:
        return question
    return rewritten.strip().strip('"')