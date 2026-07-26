"""The supervisor multi-agent graph (F9).

Wiring, exactly as the guide specifies:

    START -> supervisor
    supervisor --(plan)--> retriever | web | data | code | generate
    retriever | web | data | code -> supervisor
    generate -> critic
    critic --(verdict)--> END | revise
    revise -> supervisor

Three points worth stating explicitly, because each is a correctness trap:

1. NO STATE REDUCERS. AgentState is a plain TypedDict, so LangGraph replaces
   each returned key. That is what we want: every node already returns the
   COMPLETE new list (push_step and merge_documents merge internally). Adding
   an `operator.add` reducer would append an already-merged list to itself and
   silently duplicate every step and every document.

2. A SEPARATE "revise" NODE. The critic's "revise" verdict cannot go straight
   back to the supervisor: the supervisor refuses to re-select an agent listed
   in state["visited"], so a retry would arrive with nothing left to delegate
   to and collect no new evidence. The revise node runs reset_for_revision(),
   which reopens the agents while KEEPING the evidence already gathered.

3. THE RECURSION LIMIT IS COMPUTED, NOT GUESSED. Termination is already
   guaranteed structurally (visited caps agent hops at four; max_revisions
   caps retries). The recursion limit is the independent backstop the rubric
   requires — but set too low it would abort legitimate work instead of
   catching a runaway. required_recursion_limit() derives the true worst case
   from those two bounds, and safe_recursion_limit() never goes below it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from ai.agents.code_agent import code_agent
from ai.agents.critic import critic, reset_for_revision, route_after_critic
from ai.agents.data_sql import data_agent
from ai.agents.generate import generate_agent
from ai.agents.retriever import retriever_agent
from ai.agents.supervisor import AGENT_ROUTES, supervisor
from ai.agents.web import web_agent
from ai.config import get_settings
from ai.state import AgentState, new_state

#: Node names, in the order the graph reaches them.
SPECIALIST_NODES: Dict[str, Any] = {
    "retriever": retriever_agent,
    "web": web_agent,
    "data": data_agent,
    "code": code_agent,
}

#: Safety margin on top of the computed worst case.
RECURSION_MARGIN = 4


def required_recursion_limit(
    max_revisions: int,
    agent_count: int = len(AGENT_ROUTES),
) -> int:
    """Smallest recursion limit that cannot abort legitimate work.

    One full round, worst case: every specialist runs (agent_count hops, each
    preceded by a supervisor hop), then the supervisor says finish, then
    generate, then critic.

        per_round = agent_count * 2 + 3

    A rejected answer costs one extra `revise` hop and one more full round,
    so rounds = max_revisions + 1.
    """
    per_round = agent_count * 2 + 3
    rounds = max_revisions + 1
    return rounds * per_round + max_revisions + RECURSION_MARGIN


def safe_recursion_limit() -> int:
    """The configured limit, raised to the computed minimum if it is too low."""
    settings = get_settings()
    return max(settings.recursion_limit, required_recursion_limit(settings.max_revisions))


def route_from_supervisor(state: AgentState) -> str:
    """Read the supervisor's decision. Already validated in F7's enforce_route."""
    return state["plan"]


def build_graph(
    use_critic: bool = True,
    node_overrides: Optional[Dict[str, Any]] = None,
):
    """Compile the graph.

    use_critic=False drops the verification layer entirely (generate -> END).
    That is not a convenience: F11's rubric requires a metrics table comparing
    the system WITH and WITHOUT the critic, and this flag is how that
    comparison is produced from one code path.

    node_overrides swaps a node implementation for a stub. Used by the F9
    check to drive a deliberately mis-routing supervisor and prove the
    recursion limit stops it.
    """
    overrides = node_overrides or {}
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", overrides.get("supervisor", supervisor))
    for name, node in SPECIALIST_NODES.items():
        graph.add_node(name, overrides.get(name, node))
    graph.add_node("generate", overrides.get("generate", generate_agent))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {**{name: name for name in SPECIALIST_NODES}, "finish": "generate"},
    )
    for name in SPECIALIST_NODES:
        graph.add_edge(name, "supervisor")

    if use_critic:
        graph.add_node("critic", overrides.get("critic", critic))
        graph.add_node("revise", overrides.get("revise", reset_for_revision))
        graph.add_edge("generate", "critic")
        graph.add_conditional_edges(
            "critic",
            route_after_critic,
            {"finish": END, "revise": "revise"},
        )
        graph.add_edge("revise", "supervisor")
    else:
        graph.add_edge("generate", END)

    return graph.compile()


@lru_cache(maxsize=2)
def get_graph(use_critic: bool = True):
    """Compiled graph, cached per critic mode (compilation is not free)."""
    return build_graph(use_critic=use_critic)


def run(
    question: str,
    memory: Optional[List[str]] = None,
    use_critic: bool = True,
    callbacks: Optional[list] = None,
    recursion_limit: Optional[int] = None,
) -> AgentState:
    """Answer one question end-to-end. Never raises.

    `callbacks` is passed straight through to LangGraph — that is the hook
    Langfuse tracing (F12) plugs into, so no rewiring is needed later.

    If the recursion limit is somehow still hit, the run is reported as a
    result with an explanatory answer rather than an exception: a deployed
    backend (F14) must not return a 500 because one question mis-routed.
    """
    app = get_graph(use_critic=use_critic)
    state = new_state(question, memory=memory)

    config: Dict[str, Any] = {
        "recursion_limit": recursion_limit or safe_recursion_limit()
    }
    if callbacks:
        config["callbacks"] = callbacks

    try:
        return app.invoke(state, config=config)
    except Exception as exc:
        if "recursion" not in type(exc).__name__.lower() and "Recursion" not in str(exc):
            raise
        return {
            **state,
            "answer": (
                "The run was stopped by the step limit before an answer was "
                "verified. This means the supervisor kept routing without "
                "reaching a conclusion."
            ),
            "critic_ok": False,
            "critic_reason": f"step limit reached ({type(exc).__name__})",
            "steps": list(state.get("steps") or []) + ["ABORTED: recursion limit"],
        }


def stream(
    question: str,
    memory: Optional[List[str]] = None,
    use_critic: bool = True,
    callbacks: Optional[list] = None,
):
    """Yield (node_name, state_update) as the graph runs.

    This is what the streaming frontend (F13) consumes to show which agent is
    acting live, so the graph exposes it now rather than being reshaped later.
    """
    app = get_graph(use_critic=use_critic)
    state = new_state(question, memory=memory)

    config: Dict[str, Any] = {"recursion_limit": safe_recursion_limit()}
    if callbacks:
        config["callbacks"] = callbacks

    for chunk in app.stream(state, config=config):
        for node_name, update in chunk.items():
            yield node_name, update


def mermaid_diagram(use_critic: bool = True) -> str:
    """Mermaid source for the graph — Visual 1 of the final submission."""
    return get_graph(use_critic=use_critic).get_graph().draw_mermaid()