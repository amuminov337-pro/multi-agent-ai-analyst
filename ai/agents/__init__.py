"""Specialist and orchestrator agents for the Multi-Agent AI Analyst."""

from ai.agents.code_agent import answer_with_code, code_agent, generate_code
from ai.agents.critic import (
    critic,
    has_evidence,
    precheck,
    reset_for_revision,
    route_after_critic,
    verify,
)
from ai.agents.data_sql import answer_with_sql, data_agent, generate_sql
from ai.agents.retriever import merge_documents, retrieve, retriever_agent
from ai.agents.supervisor import (
    AGENT_ROUTES,
    VALID_ROUTES,
    available_agents,
    decide,
    enforce_route,
    supervisor,
)
from ai.agents.web import web_agent, web_search

__all__ = [
    "AGENT_ROUTES",
    "VALID_ROUTES",
    "answer_with_code",
    "answer_with_sql",
    "available_agents",
    "code_agent",
    "critic",
    "data_agent",
    "decide",
    "enforce_route",
    "generate_code",
    "generate_sql",
    "has_evidence",
    "merge_documents",
    "precheck",
    "reset_for_revision",
    "retrieve",
    "retriever_agent",
    "route_after_critic",
    "supervisor",
    "verify",
    "web_agent",
    "web_search",
]