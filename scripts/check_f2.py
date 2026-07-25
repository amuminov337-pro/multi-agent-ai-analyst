"""F2 acceptance check — Ingestion & vector store (10 pts).

Done-when (from the guide): "a document is ingested and a similarity
search returns relevant chunks."

This test is self-contained: it ingests one document carrying a unique
fact into an isolated collection, then queries for that fact and asserts
the fact comes back in the top results.
"""

from __future__ import annotations

import sys
from pathlib import Path

# make `import ai...` work when run as `python scripts/check_f2.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from ai.config import get_settings  # noqa: E402
from ai.ingestion import ingest_documents  # noqa: E402
from ai.vectorstore import (  # noqa: E402
    collection_count,
    embedding_dimension,
    get_vectorstore,
)

TEST_COLLECTION = "capstone_f2_check"

# A distinctive, unlikely-to-be-guessed fact the model can only know from
# retrieval, not from its own knowledge.
UNIQUE_FACT = "The Antares division shipped 4,271 units in fiscal year 2031."
SAMPLE_TEXT = (
    "Internal operations memo.\n\n"
    "The Borealis division focused on maintenance during the period.\n"
    f"{UNIQUE_FACT}\n"
    "The Cygnus division was paused pending review.\n"
)
QUERY = "How many units did the Antares division ship?"
NEEDLE = "4,271"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    settings = get_settings()

    section("1) EMBEDDINGS + COLLECTION")
    dim = embedding_dimension()
    print(f"embedding dimension probed : {dim}")
    print(f"qdrant mode                : {settings.qdrant_mode}")
    print(f"test collection            : {TEST_COLLECTION}")
    if dim <= 0:
        print("FAIL: embedding dimension is not positive.")
        return 1

    section("2) INGEST A DOCUMENT")
    doc = Document(page_content=SAMPLE_TEXT, metadata={"source": "f2_check"})
    n = ingest_documents([doc], collection=TEST_COLLECTION, recreate=True)
    stored = collection_count(TEST_COLLECTION)
    print(f"chunks written             : {n}")
    print(f"vectors in collection      : {stored}")
    if stored < 1:
        print("FAIL: nothing was stored in the collection.")
        return 1

    section("3) SIMILARITY SEARCH RETURNS THE FACT")
    store = get_vectorstore(collection=TEST_COLLECTION)
    results = store.similarity_search(QUERY, k=3)
    print(f"query                      : {QUERY}")
    print(f"results returned           : {len(results)}")
    for i, r in enumerate(results, 1):
        preview = r.page_content.strip().replace("\n", " ")[:80]
        print(f"  [{i}] {preview}")

    top_text = " ".join(r.page_content for r in results)
    if not results:
        print("FAIL: similarity search returned no chunks.")
        return 1
    if NEEDLE not in top_text:
        print(f"FAIL: expected '{NEEDLE}' in retrieved chunks, not found.")
        return 1

    section("RESULT")
    print("PASS — F2 done (10/10)")
    print("  - document ingested into Qdrant")
    print("  - similarity search returned the relevant chunk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())