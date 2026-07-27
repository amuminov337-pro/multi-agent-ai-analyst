"""Make RAGAS importable on current langchain versions (F11).

RAGAS 0.4.3 imports `langchain_community.chat_models.vertexai` at module load
time. That path was removed when langchain-community was sunset, so a plain
`import ragas` raises ModuleNotFoundError even though every feature we need
works fine once the import completes.

The fix is a stub registered in sys.modules BEFORE ragas is imported. The stub
classes are never instantiated — RAGAS only references them for isinstance
checks against Vertex AI clients, and we use Gemini through
LangchainLLMWrapper, so those branches are never taken.

This is isolated in its own module for one reason: no other part of the
codebase should know the workaround exists. ai/evaluation.py calls
import_ragas() once and gets a plain namespace back.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, Optional

_SHIM_INSTALLED = False


class _VertexPlaceholder:
    """Stand-in for a Vertex AI client class. Never instantiated."""


def install_ragas_shim() -> None:
    """Register stub modules for the paths RAGAS still expects. Idempotent."""
    global _SHIM_INSTALLED
    if _SHIM_INSTALLED:
        return

    chat_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    chat_vertex.ChatVertexAI = _VertexPlaceholder

    chat_models = types.ModuleType("langchain_community.chat_models")
    chat_models.vertexai = chat_vertex
    chat_models.ChatVertexAI = _VertexPlaceholder

    llms_vertex = types.ModuleType("langchain_community.llms.vertexai")
    llms_vertex.VertexAI = _VertexPlaceholder

    llms = types.ModuleType("langchain_community.llms")
    llms.vertexai = llms_vertex
    llms.VertexAI = _VertexPlaceholder

    emb_vertex = types.ModuleType("langchain_community.embeddings.vertexai")
    emb_vertex.VertexAIEmbeddings = _VertexPlaceholder

    for name, module in {
        "langchain_community.chat_models": chat_models,
        "langchain_community.chat_models.vertexai": chat_vertex,
        "langchain_community.llms": llms,
        "langchain_community.llms.vertexai": llms_vertex,
        "langchain_community.embeddings.vertexai": emb_vertex,
    }.items():
        sys.modules.setdefault(name, module)

    _SHIM_INSTALLED = True


@dataclass
class RagasBundle:
    """Everything ai/evaluation.py needs from RAGAS, resolved once."""

    version: str
    evaluate: Any
    EvaluationDataset: Any
    SingleTurnSample: Any
    faithfulness: Any
    answer_relevancy: Any
    context_precision: Any
    context_recall: Any
    LangchainLLMWrapper: Any
    LangchainEmbeddingsWrapper: Any
    RunConfig: Optional[Any]


def import_ragas() -> RagasBundle:
    """Import RAGAS behind the shim and return the pieces we use.

    Raises ImportError with a readable message if the library layout has moved
    again — better a clear failure here than a confusing one mid-evaluation.
    """
    install_ragas_shim()

    try:
        import ragas
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as exc:  # noqa: BLE001 - reported verbatim on purpose
        raise ImportError(
            f"RAGAS could not be imported even with the compatibility shim: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # RunConfig caps parallelism. Optional: without it RAGAS uses its default
    # worker count, which can trip the Gemini per-minute rate limit.
    run_config = None
    try:
        from ragas.run_config import RunConfig

        run_config = RunConfig
    except Exception:
        pass

    return RagasBundle(
        version=getattr(ragas, "__version__", "unknown"),
        evaluate=evaluate,
        EvaluationDataset=EvaluationDataset,
        SingleTurnSample=SingleTurnSample,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
        LangchainLLMWrapper=LangchainLLMWrapper,
        LangchainEmbeddingsWrapper=LangchainEmbeddingsWrapper,
        RunConfig=run_config,
    )