# Multi-Agent AI Analyst — Progress

Rubric: 100 points across 14 features (F1–F14), 5 phases.
A feature is only marked done when its "Done when" condition is demonstrated live.

| # | Feature | Phase | Points | Status | Date |
|---|---------|-------|--------|--------|------|
| F1 | Shared state & config | 1 | 5 | ✅ done | 2026-07-25 |
| F2 | Ingestion & vector store | 1 | 10 | ✅ done | 2026-07-25 |
| F3 | Retriever agent | 2 | 6 | ✅ done | 2026-07-26 |
| F4 | Web agent (Tavily) | 2 | 6 | ⬜ todo | — |
| F5 | Data agent — text-to-SQL | 2 | 10 | ⬜ todo | — |
| F6 | Code agent — Python | 2 | 8 | ⬜ todo | — |
| F7 | Supervisor / Router | 3 | 10 | ⬜ todo | — |
| F8 | Critic / Verifier | 3 | 7 | ⬜ todo | — |
| F9 | Supervisor graph (wiring) | 3 | 8 | ⬜ todo | — |
| F10 | Long-term memory | 4 | 5 | ⬜ todo | — |
| F11 | Evaluation harness | 4 | 10 | ⬜ todo | — |
| F12 | Observability (Langfuse) | 5 | 5 | ⬜ todo | — |
| F13 | Streaming frontend | 5 | 5 | ⬜ todo | — |
| F14 | Deployment | 5 | 5 | ⬜ todo | — |

**Earned so far: 21 / 100**

**Phase 1 · Foundation: ✅ complete (15/15)**

**Phase 2 · Specialist agents: F3 done, F4–F6 remaining**

## F1 — Shared state & config (5/5)

- `ai/state.py` — AgentState TypedDict (11 keys), new_state(), push_step(), evidence_bundle().
- `ai/config.py` — every secret from .env; required keys fail loudly, optional
  integrations (Tavily, Langfuse) degrade to a disabled flag.
- Environment: switched from Python 3.14 to 3.11 (venv) to avoid missing
  prebuilt wheels for ML dependencies used later (ragas, scipy).
- Model defaults updated to gemini-3.6-flash / gemini-embedding-001
  (gemini-2.5-flash and text-embedding-004 are deprecated for new users).
- Verified with `python scripts/check_f1.py` → PASS, live Gemini call succeeded.
- Git repo initialised, first commit made, `.env` confirmed excluded via `.gitignore`.

## F2 — Ingestion & vector store (10/10)

- `ai/ingestion.py` — chunk + embed + store pipeline (`ingest_documents`).
- `ai/vectorstore.py` — Qdrant collection helpers (`get_vectorstore`,
  `collection_count`, `embedding_dimension`).
- Verified with `python scripts/check_f2.py` → PASS: a document was ingested
  into an isolated Qdrant collection and a similarity search on an unrelated
  query correctly retrieved the chunk containing the planted fact.

## F3 — Retriever agent (6/6)

- `ai/agents/retriever.py` — retriever agent node.
- Verified with `python scripts/check_f3.py` → PASS.

## Final deliverables checklist (collect as we go)

- [ ] Visual 1 — supervisor multi-agent graph diagram
- [ ] Visual 2 — frontend screenshot of a live multi-agent trace
- [ ] Visual 3 — Langfuse trace of one complex question
- [ ] Visual 4 — RAGAS metrics table, with critic vs without critic
- [ ] Error analysis — 3 failures, which agent failed, one fix each
- [ ] README — diagram + metrics table + error analysis