"""Shared state contract for the multi-agent graph (F1).

Every node in the LangGraph graph reads from and writes to this single
TypedDict. Nodes never construct raw dicts by hand: they call new_state()
once at entry and push_step() on every hop, so the trace is always complete.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class AgentState(TypedDict):
    """The one object that flows through every node of the graph."""

    # --- input -------------------------------------------------------
    question: str

    # --- supervisor decision (F7) ------------------------------------
    plan: str
    visited: List[str]

    # --- evidence collected by the specialist agents (F3-F6) ---------
    documents: List[str]
    sql_result: Optional[str]
    code_result: Optional[str]

    # --- long-term memory recalled for this turn (F10) ---------------
    memory: List[str]

    # --- draft / final answer (F9) -----------------------------------
    answer: str

    # --- critic verdict (F8) -----------------------------------------
    critic_ok: bool
    critic_reason: str

    # --- control flow: guarantees termination (F9) -------------------
    steps: List[str]
    revisions: int


#: The exact key set every node must respect. Used by the F1 check script.
STATE_KEYS = frozenset(AgentState.__annotations__.keys())


def new_state(question: str, memory: Optional[List[str]] = None) -> AgentState:
    """Build a fully-initialised AgentState.

    Every key is present from the start, so no node ever hits a KeyError
    and LangGraph never has to merge a half-built dict.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    return AgentState(
        question=question.strip(),
        plan="",
        visited=[],
        documents=[],
        sql_result=None,
        code_result=None,
        memory=list(memory or []),
        answer="",
        critic_ok=False,
        critic_reason="",
        steps=[],
        revisions=0,
    )


def push_step(state: AgentState, label: str) -> List[str]:
    """Return a NEW steps list with `label` appended.

    Nodes return partial updates, so they must never mutate state in place:
        return {"documents": docs, "steps": push_step(state, "retriever")}
    """
    return list(state.get("steps", [])) + [label]


def push_visited(state: AgentState, agent: str) -> List[str]:
    """Return a NEW visited list with `agent` recorded once.

    The supervisor (F7) marks an agent as visited when it delegates to it.
    Keeping this in state — rather than trusting the LLM to remember what
    it already called — is what makes the routing loop terminate for
    structural reasons instead of behavioural ones.
    """
    visited = list(state.get("visited", []))
    if agent and agent not in visited:
        visited.append(agent)
    return visited


def evidence_bundle(state: AgentState) -> str:
    """Flatten all collected evidence into one string for the critic (F8)."""
    parts: List[str] = []
    for doc in state.get("documents", []):
        parts.append(f"[doc] {doc}")
    if state.get("sql_result"):
        parts.append(f"[sql] {state['sql_result']}")
    if state.get("code_result"):
        parts.append(f"[code] {state['code_result']}")
    for mem in state.get("memory", []):
        parts.append(f"[memory] {mem}")
    return "\n\n".join(parts) if parts else "(no evidence collected)"