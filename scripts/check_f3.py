"""F3 acceptance check — Retriever agent (6 pts).

Done-when (from the guide): "tested alone, it returns correct chunks for
a document question."

Steps:
  1. Ingest documents/ into the main capstone_docs collection (clean rebuild).
  2. Run retriever_agent ALONE on three different document questions and
     assert the expected fact is present in the retrieved chunks.
  3. Assert the AgentState contract holds: documents filled, step recorded,
     existing documents preserved (append, not overwrite).
"""

from __future__ import annotations

import sys
from pathlib import Path

# make `import ai...` work when run as `python scripts/check_f3.py`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.agents.retriever import retriever_agent  # noqa: E402
from ai.config import get_settings  # noqa: E402
from ai.ingestion import ingest_path  # noqa: E402
from ai.state import new_state  # noqa: E402
from ai.vectorstore import collection_count  # noqa: E402

DOCS_DIR = ROOT / "documents"

# (question, fact that MUST appear in the retrieved chunks)
CASES = [
    ("How many days of paid annual leave do employees get?", "24 days"),
    ("What is the hotel expense limit per night in European cities?", "180 EUR"),
    ("How fast must a severity 1 support ticket get a first response?", "1 hour"),
]


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    settings = get_settings()

    section("1) INGEST THE DOCUMENT CORPUS")
    if not DOCS_DIR.exists():
        print(f"FAIL: corpus folder not found: {DOCS_DIR}")
        return 1
    chunks = ingest_path(DOCS_DIR, recreate=True)
    stored = collection_count()
    print(f"corpus folder        : {DOCS_DIR}")
    print(f"collection           : {settings.qdrant_collection}")
    print(f"chunks written       : {chunks}")
    print(f"vectors in collection: {stored}")
    print(f"retriever k          : {settings.retriever_k}")
    if stored < 1:
        print("FAIL: corpus was not stored.")
        return 1

    section("2) RETRIEVER AGENT ALONE — THREE DOCUMENT QUESTIONS")
    failures = 0
    for question, needle in CASES:
        state = new_state(question)
        update = retriever_agent(state)
        texts = update["documents"]
        blob = " ".join(texts)
        ok = needle in blob
        print(f"\nQ: {question}")
        print(f"   chunks returned : {len(texts)}")
        print(f"   expecting       : '{needle}'")
        print(f"   result          : {'OK' if ok else 'MISSING'}")
        if texts:
            preview = texts[0].strip().replace("\n", " ")[:90]
            print(f"   top chunk       : {preview}")
        if not ok:
            failures += 1

    if failures:
        print(f"\nFAIL: {failures} of {len(CASES)} questions did not retrieve the fact.")
        return 1

    section("3) AGENTSTATE CONTRACT")
    state = new_state(CASES[0][0])
    state["documents"] = ["PRE-EXISTING CHUNK FROM ANOTHER AGENT"]
    update = retriever_agent(state)

    checks = [
        ("documents key returned", "documents" in update),
        ("steps key returned", "steps" in update),
        ("chunks retrieved", len(update["documents"]) > 1),
        (
            "existing documents preserved",
            "PRE-EXISTING CHUNK FROM ANOTHER AGENT" in update["documents"],
        ),
        (
            "step label recorded",
            any("retriever" in s for s in update["steps"]),
        ),
        (
            "respects retriever_k",
            len(update["documents"]) <= settings.retriever_k + 1,
        ),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if not all(ok for _, ok in checks):
        print("\nFAIL: AgentState contract broken.")
        return 1

    section("RESULT")
    print("PASS — F3 done (6/6)")
    print("  - corpus ingested into capstone_docs")
    print("  - retriever returns the correct chunks, tested alone")
    print("  - state updated correctly (append + step recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 