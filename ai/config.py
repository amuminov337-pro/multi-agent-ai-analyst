
"""Central configuration for the Multi-Agent AI Analyst (F1).

Reads every secret from capstone/.env — nothing is ever hard-coded and
.env is never committed. Required keys fail loudly at import time;
optional integrations (Tavily, Langfuse) degrade to a disabled flag so
the system still runs end-to-end without them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# capstone/ai/config.py -> parents[1] == capstone/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
ENV_PATH: Path = PROJECT_ROOT / ".env"
DATA_DIR: Path = PROJECT_ROOT / "data"

load_dotenv(ENV_PATH)


class ConfigError(RuntimeError):
    """Raised when a required key is missing from .env."""


def _opt(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an optional variable; treat empty/whitespace as absent."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _req(name: str) -> str:
    """Read a required variable or fail with an actionable message."""
    value = _opt(name)
    if value is None:
        raise ConfigError(
            f"{name} is missing. Add it to {ENV_PATH} "
            f"(see .env.example) and re-run."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the whole runtime configuration."""

    # --- LLM + embeddings: Gemini (required) -------------------------
    google_api_key: str
    gemini_model: str
    gemini_embed_model: str

    # --- Vector store: Qdrant cloud OR embedded ----------------------
    qdrant_url: Optional[str]
    qdrant_api_key: Optional[str]
    qdrant_path: Path
    qdrant_collection: str
    memory_collection: str

    # --- Relational DB for text-to-SQL (F5) --------------------------
    sqlite_path: Path

    # --- Optional: web search (F4) -----------------------------------
    tavily_api_key: Optional[str]

    # --- Optional: observability (F12) -------------------------------
    langfuse_public_key: Optional[str]
    langfuse_secret_key: Optional[str]
    langfuse_host: str

    # --- Graph safety limits (F9) ------------------------------------
    max_revisions: int
    recursion_limit: int

    # --- Agent tuning ------------------------------------------------
    retriever_k: int
    memory_k: int
    chunk_size: int
    chunk_overlap: int
    code_timeout_seconds: int

    @property
    def tavily_enabled(self) -> bool:
        """F4 must skip gracefully when no key is configured."""
        return self.tavily_api_key is not None

    @property
    def langfuse_enabled(self) -> bool:
        """F12 tracing is opt-in: both keys required."""
        return (
            self.langfuse_public_key is not None
            and self.langfuse_secret_key is not None
        )

    @property
    def qdrant_mode(self) -> str:
        """'cloud' when a URL is set, otherwise embedded on-disk."""
        return "cloud" if self.qdrant_url else "embedded"

    def describe(self) -> str:
        """Human-readable, secret-free summary for logs and the check script."""
        lines = [
            f"project root      : {PROJECT_ROOT}",
            f"env file          : {ENV_PATH} ({'found' if ENV_PATH.exists() else 'MISSING'})",
            f"gemini model      : {self.gemini_model}",
            f"gemini embeddings : {self.gemini_embed_model}",
            f"qdrant mode       : {self.qdrant_mode}",
            f"qdrant collection : {self.qdrant_collection}",
            f"sqlite path       : {self.sqlite_path}",
            f"tavily (F4)       : {'enabled' if self.tavily_enabled else 'disabled -> web agent will skip'}",
            f"langfuse (F12)    : {'enabled' if self.langfuse_enabled else 'disabled -> no tracing'}",
            f"max revisions     : {self.max_revisions}",
            f"recursion limit   : {self.recursion_limit}",
        ]
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and reuse the same object everywhere."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    qdrant_path = Path(_opt("QDRANT_PATH", str(DATA_DIR / "qdrant")))
    sqlite_path = Path(_opt("SQLITE_PATH", str(DATA_DIR / "company.db")))

    settings = Settings(
        google_api_key=_req("GOOGLE_API_KEY"),
        gemini_model=_opt("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        gemini_embed_model=_opt("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"),
        qdrant_url=_opt("QDRANT_URL"),
        qdrant_api_key=_opt("QDRANT_API_KEY"),
        qdrant_path=qdrant_path,
        qdrant_collection=_opt("QDRANT_COLLECTION", "capstone_docs"),
        memory_collection=_opt("MEMORY_COLLECTION", "capstone_memory"),
        sqlite_path=sqlite_path,
        tavily_api_key=_opt("TAVILY_API_KEY"),
        langfuse_public_key=_opt("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_opt("LANGFUSE_SECRET_KEY"),
        langfuse_host=_opt("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        max_revisions=int(_opt("MAX_REVISIONS", "2")),
        recursion_limit=int(_opt("RECURSION_LIMIT", "25")),
        retriever_k=int(_opt("RETRIEVER_K", "4")),
        memory_k=int(_opt("MEMORY_K", "3")),
        chunk_size=int(_opt("CHUNK_SIZE", "1000")),
        chunk_overlap=int(_opt("CHUNK_OVERLAP", "150")),
        code_timeout_seconds=int(_opt("CODE_TIMEOUT_SECONDS", "15")),
    )

    # langchain-google-genai reads this variable directly.
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    if settings.langfuse_enabled:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host

    return settings