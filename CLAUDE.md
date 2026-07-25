# Multi-Agent AI Analyst — project rules

The single source of truth is `Multi_Agent_AI_Analyst_Guide_EN.html` in this
folder (features F1–F14, 5 phases, 100 points). Read it before proposing
anything. Never deviate from it to "improve" the design.

## Workflow
- One feature at a time, in order F1 → F14. Do not start FX+1 until I confirm.
- Prototypes may be tried in `../RAG/` notebooks, but the final code lives in
  `ai/`. Never leave the same logic duplicated in both places.
- No TODOs, no placeholders, no `...` — every file must run as written.
- After each feature: update `PROGRESS.md` (status, points, date).

## Layout
- `ai/` — agent core (state, config, agents/, graph)
- `backend/` — FastAPI + SSE (F13)
- `frontend/` — Next.js (F13)
- `scripts/` — one `check_fX.py` acceptance script per feature
- `data/` — SQLite db + embedded Qdrant (git-ignored)

## Stack (fixed)
Gemini (LLM + embeddings) · Qdrant · SQLite · Tavily · Langfuse ·
LangGraph · RAGAS. Do not swap in OpenAI or any paid service.

## Hard safety requirements (points are deducted without these)
- F5 text-to-SQL: SELECT only. Reject DROP/DELETE/UPDATE/INSERT/ALTER.
  Open the DB read-only.
- F6 code agent: run model-written Python in a sandbox with a timeout cap.
- F9 graph: enforce `recursion_limit` and `max_revisions` (max 2) so the
  critic loop always terminates.
- F11 eval harness: must support a flag to run with and without the critic —
  the rubric requires a comparison table.
- F4 web agent: if `TAVILY_API_KEY` is absent, skip gracefully — never raise.

## Secrets
All keys live in `.env` (git-ignored). `.env.example` is committed with empty
values. Never print a full key to stdout; mask it.

## Errors
When something fails, ask me for the full traceback instead of guessing.