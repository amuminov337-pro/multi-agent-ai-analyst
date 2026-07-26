"""Specialist and orchestrator agents for the Multi-Agent AI Analyst."""

from ai.agents.code_agent import answer_with_code, code_agent, generate_code
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
    "data_agent",
    "decide",
    "enforce_route",
    "generate_code",
    "generate_sql",
    "merge_documents",
    "retrieve",
    "retriever_agent",
    "supervisor",
    "web_agent",
    "web_search",
]