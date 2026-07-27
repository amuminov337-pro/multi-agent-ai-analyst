"""
F13 - FastAPI backend, SSE streaming.

Nima uchun thread + navbat: ai.graph.stream() sinxron generator, FastAPI esa
async. Generatorni to'g'ridan-to'g'ri async funksiyada aylantirsak, event loop
bloklanadi va barcha qadamlar oxirida birdan chiqadi - ya'ni "streaming"
ko'rinishi yo'qoladi. Shuning uchun graf alohida thread'da yuradi, har bir
chunk asyncio navbatiga tushadi va zudlik bilan mijozga jo'natiladi.

MUHIM: ai.graph.stream() (node_name, update) TUPLE qaytaradi, dict emas.
_normalize() aynan shu shaklga moslangan - aks holda har bir hodisa matnga
aylanib, tugun nomi ham, yakuniy javob ham yo'qoladi.

Endpointlar:
    GET  /api/health           - tiriklik + tracing holati
    GET  /api/stream?question= - SSE, jonli agent qadamlari (EventSource shu yerga ulanadi)
    POST /api/ask              - oddiy JSON javob (streaming'siz)
    GET  /api/graph            - mermaid diagramma matni

Ishga tushirish (repo ildizidan, venv faol):
    uvicorn backend.app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ai import observability as obs

HEARTBEAT_SECONDS = 15.0

app = FastAPI(title="Multi-Agent AI Analyst", version="1.0.0")

_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


# ----------------------------------------------------------------- yordamchi

def _handler_trace_id(handler: Any) -> Optional[str]:
    """Callback handler'dan trace id (F12 bilan bir xil mantiq, lokal nusxa)."""
    if handler is None:
        return None
    for attr in ("last_trace_id", "trace_id"):
        value = getattr(handler, attr, None)
        if isinstance(value, str) and value:
            return value
    getter = getattr(handler, "get_trace_id", None)
    if callable(getter):
        try:
            value = getter()
            if isinstance(value, str) and value:
                return value
        except Exception:  # noqa: BLE001
            pass
    return None


def _shorten(value: Any, limit: int = 400) -> Any:
    """Uzun oraliq qiymatni SSE uchun qisqartiradi (yakuniy javobga qo'llanmaydi)."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + " ..."
    if isinstance(value, list):
        return [_shorten(v, limit) for v in value[:20]]
    return value


def _node_event(node: Optional[str], payload: Any) -> Dict[str, Any]:
    """Bitta tugun yangilanishini SSE hodisasiga aylantiradi."""
    event: Dict[str, Any] = {"type": "step", "node": node}

    if not isinstance(payload, dict):
        event["text"] = _shorten(str(payload))
        return event

    # answer QISQARTIRILMAYDI - frontend to'liq javobni ko'rsatishi kerak
    if "answer" in payload:
        event["answer"] = payload.get("answer")

    for key in ("steps", "visited", "critic_ok", "critic_reason", "revisions", "plan"):
        if key in payload:
            event[key] = _shorten(payload.get(key))

    if "documents" in payload:
        docs = payload.get("documents") or []
        event["documents"] = len(docs) if isinstance(docs, list) else None
    if "sql_result" in payload:
        event["sql_result"] = _shorten(payload.get("sql_result"))
    if "code_result" in payload:
        event["code_result"] = _shorten(payload.get("code_result"))

    return event


def _normalize(chunk: Any) -> List[Dict[str, Any]]:
    """
    ai.graph.stream() chunk'ini SSE hodisalariga aylantiradi.

    Asosiy shakl: (node_name, update) tuple. Qolgan shakllar zaxira sifatida
    qo'llanadi, shunda graph.py kelajakda o'zgarsa ham backend sinmaydi.
    """
    # 1) Asosiy shakl: tuple
    if isinstance(chunk, tuple) and len(chunk) == 2:
        node, payload = chunk
        return [_node_event(str(node), payload)]

    # 2) Zaxira: {node: update} dict
    if isinstance(chunk, dict):
        if "steps" in chunk or "answer" in chunk:
            return [_node_event(None, chunk)]
        return [_node_event(str(node), payload) for node, payload in chunk.items()]

    # 3) Zaxira: oddiy matn
    return [{"type": "step", "node": None, "text": _shorten(str(chunk))}]


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------ endpointlar

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "tracing": obs.is_enabled(),
        "langfuse_sdk": obs.sdk_version(),
    }


@app.get("/api/graph")
def graph_diagram() -> Dict[str, Any]:
    try:
        from ai.graph import mermaid_diagram
        return {"mermaid": mermaid_diagram()}
    except Exception as exc:  # noqa: BLE001
        return {"mermaid": None, "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/ask")
def ask(request: AskRequest) -> JSONResponse:
    """Streaming'siz to'liq javob - frontend fallback'i va tez sinov uchun."""
    result = obs.run_traced(request.question, trace_name="api-ask")
    state = result["state"]
    return JSONResponse(
        {
            "answer": state.get("answer"),
            "steps": state.get("steps"),
            "visited": state.get("visited"),
            "critic_ok": state.get("critic_ok"),
            "trace_url": result.get("url"),
        }
    )


async def _event_stream(question: str) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    started = time.time()

    def put(item: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def worker() -> None:
        handler = None
        try:
            from ai.graph import stream as graph_stream

            callbacks = obs.tracing_callbacks()
            handler = callbacks[0] if callbacks else None
            for chunk in graph_stream(question, callbacks=callbacks):
                put(("chunk", chunk))
        except Exception as exc:  # noqa: BLE001
            put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            trace_id = _handler_trace_id(handler)
            try:
                obs.flush()
            except Exception:  # noqa: BLE001
                pass
            put(("done", obs.trace_url(trace_id) if trace_id else ""))

    threading.Thread(target=worker, daemon=True).start()

    yield _sse("start", {"type": "start", "question": question})

    last_state: Dict[str, Any] = {}

    while True:
        try:
            kind, payload = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
        except asyncio.TimeoutError:
            # Proxy'lar jim turgan ulanishni uzib yuboradi - izoh qatori uni tirik saqlaydi
            yield ": keepalive\n\n"
            continue

        if kind == "chunk":
            for event in _normalize(payload):
                event["elapsed"] = round(time.time() - started, 2)
                if event.get("answer"):
                    last_state["answer"] = event["answer"]
                if event.get("steps"):
                    last_state["steps"] = event["steps"]
                if event.get("critic_ok") is not None:
                    last_state["critic_ok"] = event["critic_ok"]
                yield _sse("step", event)

        elif kind == "error":
            yield _sse("error", {"type": "error", "message": payload})

        elif kind == "done":
            yield _sse(
                "done",
                {
                    "type": "done",
                    "answer": last_state.get("answer"),
                    "steps": last_state.get("steps"),
                    "critic_ok": last_state.get("critic_ok"),
                    "trace_url": payload,
                    "elapsed": round(time.time() - started, 2),
                },
            )
            return


@app.get("/api/stream")
async def stream_endpoint(question: str = Query(..., min_length=3)) -> StreamingResponse:
    """SSE oqimi - brauzerdagi EventSource shu manzilga ulanadi."""
    return StreamingResponse(
        _event_stream(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx buferlashini o'chiradi (F14 deploy uchun muhim)
        },
    )