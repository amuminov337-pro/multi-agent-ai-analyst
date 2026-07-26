# Multi-Agent AI Analyst — Progress

Rubric: 100 points across 14 features (F1–F14), 5 phases.
A feature is only marked done when its "Done when" condition is demonstrated live.

| # | Feature | Phase | Points | Status | Date |
|---|---------|-------|--------|--------|------|
| F1 | Shared state & config | 1 | 5 | ✅ done | 2026-07-25 |
| F2 | Ingestion & vector store | 1 | 10 | ✅ done | 2026-07-25 |
| F3 | Retriever agent | 2 | 6 | ✅ done | 2026-07-26 |
| F4 | Web agent (Tavily) | 2 | 6 | ✅ done | 2026-07-26 |
| F5 | Data agent — text-to-SQL | 2 | 10 | ✅ done | 2026-07-26 |
| F6 | Code agent — Python | 2 | 8 | ✅ done | 2026-07-26 |
| F7 | Supervisor / Router | 3 | 10 | ✅ done | 2026-07-26 |
| F8 | Critic / Verifier | 3 | 7 | ✅ done | 2026-07-26 |
| F9 | Supervisor graph (wiring) | 3 | 8 | ✅ done | 2026-07-26 |
| F10 | Long-term memory | 4 | 5 | ⬜ todo | — |
| F11 | Evaluation harness | 4 | 10 | ⬜ todo | — |
| F12 | Observability (Langfuse) | 5 | 5 | ⬜ todo | — |
| F13 | Streaming frontend | 5 | 5 | ⬜ todo | — |
| F14 | Deployment | 5 | 5 | ⬜ todo | — |

**Earned so far: 70 / 100**

**Phase 1 · Foundation: ✅ complete (15/15)**

**Phase 2 · Specialist agents: ✅ complete (30/30)**

**Phase 3 · Orchestration: ✅ complete (25/25)**

## F1 — Shared state & config (5/5)

- `ai/state.py` — AgentState TypedDict (11 keys), new_state(), push_step(), evidence_bundle().
- `ai/config.py` — every secret from .env; required keys fail loudly, optional
  integrations (Tavily, Langfuse) degrade to a disabled flag.
- Environment: switched from Python 3.14 to 3.11 (venv) to avoid missing
  prebuilt wheels for ML dependencies used later (ragas, scipy).
- Model defaults updated to gemini-3.1-flash-lite / gemini-embedding-001
  (gemini-2.5-flash hit a 404 as deprecated, and gemini-3.6-flash's free-tier
  quota was too low; gemini-3.1-flash-lite is a stable, probed-working name).
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

## F4 — Web agent (6/6)

- `ai/agents/web.py` — Tavily-backed web agent node; skips gracefully with
  no exception when `TAVILY_API_KEY` is absent.
- Verified with `python scripts/check_f4.py` → PASS (live search + graceful
  skip both confirmed).

## F5 — Data agent / Text-to-SQL (10/10)

- `ai/agents/data_sql.py` — text-to-SQL agent; SELECT-only guard rejects
  DROP/DELETE/UPDATE/INSERT/ALTER/PRAGMA/ATTACH, multi-statement and
  comment-smuggling attempts; DB opened read-only at the OS level.
- Verified with `python scripts/check_f5.py` → PASS: correct answers on 3
  ground-truth questions, 10/10 malicious SQL attacks rejected, OS-level
  read-only connection confirmed (write blocked by SQLite itself).

## F6 — Code agent (8/8)

- `ai/agents/code_agent.py` — sandboxed Python code agent with subprocess
  isolation and a hard runtime cap.
- Verified with `python scripts/check_f6.py` → PASS: 13/13 sandbox attacks
  rejected, runtime cap killed an infinite loop at 3s, no secrets leaked
  into the child process.

## F7 — Supervisor / Router: ✅ DONE (10/10 points), sana: 2026-07-26, deterministic guard: 14/14 test o'tdi, visited-tracking orqali takroriy routing kod darajasida bloklandi, LLM model gemini-3.1-flash-lite ga o'zgartirildi (kvota muammosi tufayli)

## F8 — Critic / Verifier: ✅ DONE (7/7 points), sana: 2026-07-26, deterministic pre-check + LLM grounding tekshiruvi, 3/3 buzuq javob ushlandi (noto'g'ri raqam, o'ylab topilgan fakt, mavzudan chetga chiqish), revision cap ishladi

## F9 — Supervisor graph wiring: ✅ DONE (8/8 points), sana: 2026-07-26, eslatma: to'rtta agent + supervisor + critic to'liq bog'landi, birinchi end-to-end multi-part savol muvaffaqiyatli o'tdi, mis-routing loop recursion limit bilan to'xtatilishi tasdiqlandi

## Final deliverables checklist (collect as we go)

- [ ] Visual 1 — supervisor multi-agent graph diagram
- [ ] Visual 2 — frontend screenshot of a live multi-agent trace
- [ ] Visual 3 — Langfuse trace of one complex question
- [ ] Visual 4 — RAGAS metrics table, with critic vs without critic
- [ ] Error analysis — 3 failures, which agent failed, one fix each
- [ ] README — diagram + metrics table + error analysis