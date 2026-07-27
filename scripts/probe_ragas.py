"""Probe whether RAGAS can be made importable and usable with Gemini (F11).

RAGAS 0.4.3 imports `langchain_community.chat_models.vertexai`, a path that no
longer exists in current langchain-community (the package is sunset). This
script tests two things empirically instead of guessing:

  A. Does a plain `import ragas` work?
  B. If not, does injecting a stub module into sys.modules make it importable,
     and are the metrics and Gemini wrappers then reachable?

Nothing here is a permanent fix — it only tells us which path F11 can take.

    python scripts/probe_ragas.py
"""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def try_plain_import() -> bool:
    print("A) PLAIN IMPORT")
    try:
        import ragas  # noqa: F401

        print(f"   [OK] import ragas succeeded (version {ragas.__version__})")
        return True
    except Exception as exc:
        print(f"   [FAIL] {type(exc).__name__}: {exc}")
        return False


def install_shim() -> None:
    """Register a stub for the module path RAGAS still expects."""
    for name in list(sys.modules):
        if name.startswith("ragas"):
            del sys.modules[name]

    class ChatVertexAI:  # placeholder: only used in isinstance checks
        pass

    class VertexAI:
        pass

    class VertexAIEmbeddings:
        pass

    chat_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    chat_vertex.ChatVertexAI = ChatVertexAI

    chat_models = types.ModuleType("langchain_community.chat_models")
    chat_models.vertexai = chat_vertex
    chat_models.ChatVertexAI = ChatVertexAI

    llms_vertex = types.ModuleType("langchain_community.llms.vertexai")
    llms_vertex.VertexAI = VertexAI

    llms = types.ModuleType("langchain_community.llms")
    llms.vertexai = llms_vertex
    llms.VertexAI = VertexAI

    emb_vertex = types.ModuleType("langchain_community.embeddings.vertexai")
    emb_vertex.VertexAIEmbeddings = VertexAIEmbeddings

    for name, module in {
        "langchain_community.chat_models": chat_models,
        "langchain_community.chat_models.vertexai": chat_vertex,
        "langchain_community.llms": llms,
        "langchain_community.llms.vertexai": llms_vertex,
        "langchain_community.embeddings.vertexai": emb_vertex,
    }.items():
        sys.modules.setdefault(name, module)


def try_shimmed_import() -> bool:
    print("\nB) IMPORT WITH A sys.modules SHIM")
    install_shim()
    try:
        import ragas

        print(f"   [OK] import ragas succeeded (version {ragas.__version__})")
    except Exception as exc:
        print(f"   [FAIL] import ragas: {type(exc).__name__}: {exc}")
        return False

    ok = True
    checks = [
        ("from ragas import evaluate", "ragas", "evaluate"),
    ]
    for label, module_name, attr in checks:
        try:
            module = __import__(module_name, fromlist=[attr])
            getattr(module, attr)
            print(f"   [OK] {label}")
        except Exception as exc:
            print(f"   [FAIL] {label}: {type(exc).__name__}: {exc}")
            ok = False

    print("\n   metrics:")
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        try:
            module = __import__("ragas.metrics", fromlist=[metric])
            getattr(module, metric)
            print(f"     [OK] {metric}")
        except Exception as exc:
            print(f"     [FAIL] {metric}: {type(exc).__name__}")
            ok = False

    print("\n   Gemini wrappers:")
    for module_name, attr in (
        ("ragas.llms", "LangchainLLMWrapper"),
        ("ragas.embeddings", "LangchainEmbeddingsWrapper"),
    ):
        try:
            module = __import__(module_name, fromlist=[attr])
            getattr(module, attr)
            print(f"     [OK] {module_name}.{attr}")
        except Exception as exc:
            print(f"     [FAIL] {module_name}.{attr}: {type(exc).__name__}: {exc}")
            ok = False

    print("\n   dataset entry point:")
    for module_name, attr in (
        ("ragas", "EvaluationDataset"),
        ("ragas", "SingleTurnSample"),
    ):
        try:
            module = __import__(module_name, fromlist=[attr])
            getattr(module, attr)
            print(f"     [OK] {module_name}.{attr}")
        except Exception:
            print(f"     [--] {module_name}.{attr} not available (may use dict/Dataset API)")

    return ok


def main() -> int:
    print("=" * 60)
    print("RAGAS COMPATIBILITY PROBE")
    print("=" * 60)

    if try_plain_import():
        print("\nVERDICT: RAGAS works as installed — no shim needed.")
        return 0

    if try_shimmed_import():
        print("\nVERDICT: RAGAS works WITH the shim. F11 can use the real library.")
        return 0

    print("\nVERDICT: RAGAS is not usable with these package versions.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
