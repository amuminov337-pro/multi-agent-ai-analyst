"""Supervisor / Router — the manager of the multi-agent system (F7).

Reads the question plus everything gathered so far, and decides which
specialist runs next — or that enough evidence exists and the graph should
move on to drafting the answer ("finish").

The LLM's choice is a SUGGESTION, never an instruction. It always passes
through enforce_route(), a pure deterministic function that:

  * normalises the label (case, whitespace, stray punctuation),
  * rejects anything outside the known route set,
  * refuses to re-select an agent already recorded in state["visited"],
  * returns "finish" once every specialist has run.

That is what makes termination a structural property rather than a
behavioural hope: even a model that keeps answering "data" forever cannot
produce more than four agent hops. The F9 recursion limit then remains a
second, independent backstop rather than the only one.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from ai.llm import get_llm, response_text
from ai.state import AgentState, push_step, push_visited

#: Specialists the supervisor may delegate to.
AGENT_ROUTES: Tuple[str, ...] = ("retriever", "web", "data", "code")
#: Every legal decision, including the terminal one.
VALID_ROUTES: Tuple[str, ...] = AGENT_ROUTES + ("finish",)
#: Used when the very first decision is unusable — documents are the
#: cheapest, safest place to start looking.
DEFAULT_FIRST_ROUTE = "retriever"

SUPERVISOR_PROMPT = """You are the supervisor of a team of specialist agents.
Decide which ONE agent should run next, or answer "finish".

The agents:
- retriever : searches the internal company documents (handbook, policies,
              service levels, security rules). Use it for questions about
              rules, policies, definitions or explanations.
- web       : live web search. Use it ONLY for public, external or current
              information that internal documents cannot contain.
- data      : writes and runs SQL against the company analytics database
              (employees, departments, customers, support_tickets, expenses,
              leave_requests). Use it for counts, sums, averages, rankings
              and any question about company records.
- code      : writes and runs Python. Use it for pure calculation,
              arithmetic, combinatorics or transforming numbers already
              collected.
- finish    : enough evidence has been gathered to answer the question.

Question: {question}

Agents already used (you may NOT choose these again): {visited}
Agents still available: {available}

Evidence collected so far:
{evidence}

Rules:
- Choose exactly one label from: {available_or_finish}
- Never choose an agent listed as already used.
- Choose "finish" as soon as the evidence is enough to answer the question.
- A question needing both a number and an explanation needs both the data
  agent and the retriever — route to one now, the other on the next turn.

Answer with the single label only."""


class Route(BaseModel):
    """Structured decision returned by the supervisor LLM."""

    next: str = Field(description="One of: retriever, web, data, code, finish")
    reason: str = Field(default="", description="One short sentence of justification")


def available_agents(state: AgentState) -> List[str]:
    """Specialists that have not been used yet in this run."""
    visited = state.get("visited") or []
    return [agent for agent in AGENT_ROUTES if agent not in visited]


def _evidence_preview(state: AgentState, limit: int = 700) -> str:
    """Short digest of what has been collected, to keep the prompt cheap."""
    parts: List[str] = []
    docs = state.get("documents") or []
    if docs:
        parts.append(f"- documents: {len(docs)} chunk(s) retrieved")
    if state.get("sql_result"):
        parts.append(f"- sql: {str(state['sql_result'])[:limit]}")
    if state.get("code_result"):
        parts.append(f"- code: {str(state['code_result'])[:limit]}")
    memory = state.get("memory") or []
    if memory:
        parts.append(f"- memory: {len(memory)} earlier turn(s) recalled")
    return "\n".join(parts) if parts else "(nothing collected yet)"


def enforce_route(raw_choice: Optional[str], state: AgentState) -> Tuple[str, str]:
    """Turn a raw LLM label into a legal route. Pure and deterministic.

    Returns (route, note). `note` is empty when the model's choice was
    accepted as-is, and otherwise explains the correction — that string is
    written into the trace so a mis-routing model stays visible in Langfuse
    (F12) instead of being silently patched over.
    """
    available = available_agents(state)
    choice = (raw_choice or "").strip().strip(".\"'`").lower()
    # tolerate "supervisor -> data" or "route: data"
    if choice:
        choice = choice.split()[-1].split(">")[-1].split(":")[-1].strip()

    # Nothing left to delegate to: the run is structurally over.
    if not available:
        note = "" if choice == "finish" else "no agents left, forced finish"
        return "finish", note

    if choice not in VALID_ROUTES:
        if state.get("visited"):
            return "finish", f"unusable choice {raw_choice!r}, forced finish"
        return (
            DEFAULT_FIRST_ROUTE,
            f"unusable choice {raw_choice!r}, defaulted to {DEFAULT_FIRST_ROUTE}",
        )

    if choice == "finish":
        return "finish", ""

    if choice not in available:
        return "finish", f"'{choice}' already used, forced finish"

    return choice, ""


def decide(state: AgentState) -> Tuple[str, str]:
    """Ask the LLM for the next route, then enforce it. Never raises."""
    available = available_agents(state)
    if not available:
        return "finish", "no agents left, forced finish"

    prompt = SUPERVISOR_PROMPT.format(
        question=state["question"],
        visited=", ".join(state.get("visited") or []) or "(none)",
        available=", ".join(available),
        available_or_finish=", ".join(available + ["finish"]),
        evidence=_evidence_preview(state),
    )

    try:
        structured = get_llm().with_structured_output(Route)
        decision = structured.invoke(prompt)
        raw = decision.next
    except Exception:
        # Structured output can fail on model or schema changes; plain text
        # still carries the label, so fall back rather than break the graph.
        try:
            raw = response_text(get_llm().invoke(prompt))
        except Exception as exc:
            return "finish", f"supervisor LLM failed ({type(exc).__name__}), forced finish"

    return enforce_route(raw, state)


def supervisor(state: AgentState) -> dict:
    """LangGraph node: pick the next agent, or finish.

    Writes the decision to state["plan"], records the delegation in
    state["visited"], and appends a readable hop to state["steps"].
    """
    route, note = decide(state)

    label = f"supervisor→{route}"
    if note:
        label += f" [{note}]"

    update = {
        "plan": route,
        "steps": push_step(state, label),
    }
    if route != "finish":
        update["visited"] = push_visited(state, route)
    return update