"""Document ingestion for the Multi-Agent AI Analyst (F2).

Pipeline (straight from the guide):
    load documents -> chunk (1000 / 150) -> embed (Gemini) -> store (Qdrant)

Supports a single file or a whole directory (recursively). File type is
picked by extension; anything unknown is read as plain UTF-8 text.

CLI:
    python -m ai.ingestion path/to/file_or_dir
    python -m ai.ingestion path/to/dir --recreate      # wipe collection first
    python -m ai.ingestion path/to/dir --collection my_docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai.config import get_settings
from ai.vectorstore import collection_count, get_vectorstore


def _loader_for(path: Path):
    """Choose a LangChain loader based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix in {".html", ".htm"}:
        return BSHTMLLoader(str(path))
    if suffix == ".csv":
        return CSVLoader(str(path))
    # .txt, .md, .rst, .json, unknown -> plain text
    return TextLoader(str(path), encoding="utf-8")


def _iter_files(path: Path) -> List[Path]:
    """Expand a path into a list of concrete files."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    raise FileNotFoundError(f"Path not found: {path}")


def load_documents(path: str | Path) -> List[Document]:
    """Load one file or a directory into LangChain Documents."""
    root = Path(path).expanduser().resolve()
    docs: List[Document] = []
    for file_path in _iter_files(root):
        try:
            docs.extend(_loader_for(file_path).load())
        except Exception as exc:  # keep going; report the offending file
            print(f"  ! skipped {file_path.name}: {exc}", file=sys.stderr)
    if not docs:
        raise ValueError(f"No readable documents found under {root}")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents with the guide's chunk size / overlap."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def ingest_documents(
    docs: List[Document],
    collection: Optional[str] = None,
    recreate: bool = False,
) -> int:
    """Chunk + embed + store already-loaded Documents. Returns chunk count."""
    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError("Nothing to ingest after chunking.")
    store = get_vectorstore(collection=collection, recreate=recreate)
    store.add_documents(chunks)
    return len(chunks)


def ingest_path(
    path: str | Path,
    collection: Optional[str] = None,
    recreate: bool = False,
) -> int:
    """Full pipeline for a file/dir path. Returns the number of chunks stored."""
    docs = load_documents(path)
    return ingest_documents(docs, collection=collection, recreate=recreate)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant (F2).")
    parser.add_argument("path", help="File or directory to ingest.")
    parser.add_argument("--collection", default=None, help="Target collection name.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild the collection before ingesting.",
    )
    args = parser.parse_args()

    settings = get_settings()
    name = args.collection or settings.qdrant_collection

    print(f"Ingesting: {args.path}")
    print(f"Collection: {name} (recreate={args.recreate})")
    n = ingest_path(args.path, collection=args.collection, recreate=args.recreate)
    total = collection_count(name)
    print(f"Stored {n} chunks. Collection now holds {total} vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())