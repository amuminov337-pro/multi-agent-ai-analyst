"""
F12 - Langfuse observability.

Grafni qayta simlash SHART EMAS: ai/graph.py dagi run()/stream() allaqachon
`callbacks` parametrini LangGraph'ga uzatadi. Bu modul faqat:
  - Langfuse callback handler yaratadi (SDK 2.x / 3.x / 4.x farqini yashiradi),
  - run'ni tracing bilan ishga tushiradi va trace_id qaytaradi,
  - trace'ni API orqali qayta o'qib observation va token sonini tasdiqlaydi.

MUHIM: Langfuse eventlarni ASINXRON indekslaydi. Trace yozuvi bir soniyada
paydo bo'ladi, ichidagi observation'lar esa 10-30 soniyadan keyin. Shuning
uchun fetch_trace() bo'sh javobni ham "hali tayyor emas" deb hisoblab,
observation'lar kelguncha qayta so'raydi.

Kalit yo'q bo'lsa modul XATO BERMAYDI - F4 dagi Tavily bilan bir xil qoida:
tracing o'chadi, graf ishlashda davom etadi.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from ai.config import get_settings

DEFAULT_HOST = "https://cloud.langfuse.com"

_SDK: Dict[str, Any] = {"handler_cls": None, "major": None, "import_error": None}
_CLIENT: Dict[str, Any] = {"client": None}


class ObservabilityError(RuntimeError):
    """Langfuse bilan bog'liq, tiklab bo'lmaydigan xato."""


# ---------------------------------------------------------------- SDK yuklash

def load_sdk() -> Dict[str, Any]:
    """Langfuse CallbackHandler klassini topadi. 3.x/4.x -> 2.x tartibida."""
    if _SDK["handler_cls"] is not None or _SDK["import_error"] is not None:
        return _SDK

    errors: List[str] = []

    try:
        from langfuse.langchain import CallbackHandler  # type: ignore
        _SDK["handler_cls"] = CallbackHandler
        _SDK["major"] = 3  # 3.x va 4.x bir xil API
        return _SDK
    except Exception as exc:  # noqa: BLE001
        errors.append(f"langfuse.langchain -> {type(exc).__name__}: {exc}")

    try:
        from langfuse.callback import CallbackHandler  # type: ignore
        _SDK["handler_cls"] = CallbackHandler
        _SDK["major"] = 2
        return _SDK
    except Exception as exc:  # noqa: BLE001
        errors.append(f"langfuse.callback -> {type(exc).__name__}: {exc}")

    _SDK["import_error"] = " | ".join(errors)
    return _SDK


def sdk_version() -> str:
    try:
        from importlib.metadata import version
        return version("langfuse")
    except Exception:  # noqa: BLE001
        return "unknown"


# ------------------------------------------------------------- Kalitlar / env

def _cred(settings: Any, attr: str, env_key: str) -> str:
    value = getattr(settings, attr, None) or os.getenv(env_key, "")
    return str(value or "").strip()


def credentials() -> Dict[str, str]:
    settings = get_settings()
    return {
        "public_key": _cred(settings, "langfuse_public_key", "LANGFUSE_PUBLIC_KEY"),
        "secret_key": _cred(settings, "langfuse_secret_key", "LANGFUSE_SECRET_KEY"),
        "host": _cred(settings, "langfuse_host", "LANGFUSE_HOST") or DEFAULT_HOST,
    }


def is_enabled() -> bool:
    creds = credentials()
    if not creds["public_key"] or not creds["secret_key"]:
        return False
    return load_sdk()["handler_cls"] is not None


def _export_env() -> Dict[str, str]:
    creds = credentials()
    if creds["public_key"]:
        os.environ["LANGFUSE_PUBLIC_KEY"] = creds["public_key"]
    if creds["secret_key"]:
        os.environ["LANGFUSE_SECRET_KEY"] = creds["secret_key"]
    if creds["host"]:
        os.environ["LANGFUSE_HOST"] = creds["host"]
    return creds


# ------------------------------------------------------------------- Klient

def get_client() -> Any:
    if _CLIENT["client"] is not None:
        return _CLIENT["client"]

    sdk = load_sdk()
    if sdk["handler_cls"] is None:
        raise ObservabilityError(f"Langfuse SDK import bo'lmadi: {sdk['import_error']}")

    creds = _export_env()
    if not creds["public_key"] or not creds["secret_key"]:
        raise ObservabilityError("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY topilmadi (.env).")

    if sdk["major"] == 3:
        try:
            from langfuse import get_client as _get_client  # type: ignore
            client = _get_client()
        except Exception:  # noqa: BLE001
            from langfuse import Langfuse  # type: ignore
            client = Langfuse(
                public_key=creds["public_key"],
                secret_key=creds["secret_key"],
                host=creds["host"],
            )
    else:
        from langfuse import Langfuse  # type: ignore
        client = Langfuse(
            public_key=creds["public_key"],
            secret_key=creds["secret_key"],
            host=creds["host"],
        )

    _CLIENT["client"] = client
    return client


def auth_check() -> bool:
    client = get_client()
    checker = getattr(client, "auth_check", None)
    if not callable(checker):
        return True
    return bool(checker())


# --------------------------------------------------------------- Callback

def get_callback_handler(**kwargs: Any) -> Any:
    sdk = load_sdk()
    if sdk["handler_cls"] is None:
        raise ObservabilityError(f"Langfuse SDK import bo'lmadi: {sdk['import_error']}")

    creds = _export_env()
    if sdk["major"] == 3:
        get_client()
        return sdk["handler_cls"](**kwargs)

    return sdk["handler_cls"](
        public_key=creds["public_key"],
        secret_key=creds["secret_key"],
        host=creds["host"],
        **kwargs,
    )


def tracing_callbacks() -> List[Any]:
    """Tayyor callbacks ro'yxati; tracing o'chiq bo'lsa bo'sh ro'yxat."""
    if not is_enabled():
        return []
    try:
        return [get_callback_handler()]
    except ObservabilityError:
        return []


def flush() -> None:
    """Buferdagi eventlarni serverga yuboradi (skript tugashidan oldin shart)."""
    try:
        client = get_client()
    except ObservabilityError:
        return
    fn = getattr(client, "flush", None)
    if callable(fn):
        try:
            fn()
            return
        except Exception:  # noqa: BLE001
            pass
    fn = getattr(client, "shutdown", None)
    if callable(fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass


def _handler_trace_id(handler: Any) -> Optional[str]:
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


def trace_url(trace_id: Optional[str]) -> str:
    if not trace_id:
        return ""
    try:
        client = get_client()
        getter = getattr(client, "get_trace_url", None)
        if callable(getter):
            try:
                url = getter(trace_id=trace_id)
            except TypeError:
                url = getter()
            if url:
                return str(url)
    except Exception:  # noqa: BLE001
        pass
    host = credentials()["host"].rstrip("/")
    return f"{host}/trace/{trace_id}"


# ------------------------------------------------------------- Tracing bilan run

def run_traced(
    question: str,
    *,
    trace_name: str = "multi-agent-ai-analyst",
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **run_kwargs: Any,
) -> Dict[str, Any]:
    """
    Grafni Langfuse tracing bilan ishga tushiradi.
    Qaytaradi: {"state": AgentState, "trace_id": str|None, "url": str, "traced": bool}
    """
    from ai.graph import run as graph_run  # kech import - aylanma importni oldini oladi

    if not is_enabled():
        state = graph_run(question, callbacks=[], **run_kwargs)
        return {"state": state, "trace_id": None, "url": "", "traced": False}

    handler = get_callback_handler()
    client = get_client()
    sdk = load_sdk()
    trace_id: Optional[str] = None

    if sdk["major"] == 3 and hasattr(client, "start_as_current_span"):
        with client.start_as_current_span(name=trace_name) as _span:
            try:
                trace_id = client.get_current_trace_id()
            except Exception:  # noqa: BLE001
                trace_id = None
            try:
                client.update_current_trace(
                    input={"question": question},
                    session_id=session_id,
                    tags=tags or ["capstone", "F12"],
                )
            except Exception:  # noqa: BLE001
                pass

            state = graph_run(question, callbacks=[handler], **run_kwargs)

            try:
                client.update_current_trace(output={"answer": state.get("answer")})
            except Exception:  # noqa: BLE001
                pass
    else:
        state = graph_run(question, callbacks=[handler], **run_kwargs)
        trace_id = _handler_trace_id(handler)

    if not trace_id:
        trace_id = _handler_trace_id(handler)

    flush()
    return {
        "state": state,
        "trace_id": trace_id,
        "url": trace_url(trace_id),
        "traced": True,
    }


# --------------------------------------------------- Trace'ni qayta o'qish

def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, dict):
                    return data
            except Exception:  # noqa: BLE001
                continue
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _trace_observations(trace: Any) -> List[Any]:
    data = _as_dict(trace)
    items = data.get("observations")
    return list(items) if items else []


def _get_trace_raw(client: Any, trace_id: str) -> Any:
    api = getattr(client, "api", None)
    if api is not None and hasattr(api, "trace"):
        return api.trace.get(trace_id)
    getter = getattr(client, "get_trace", None)
    if callable(getter):
        return getter(trace_id)
    raise ObservabilityError("SDK'da trace o'qish metodi topilmadi.")


def _list_observations(client: Any, trace_id: str) -> List[Any]:
    """Ikkinchi yo'l: observation'lar alohida endpoint orqali."""
    api = getattr(client, "api", None)
    if api is None or not hasattr(api, "observations"):
        return []
    listing = api.observations.get_many(trace_id=trace_id)
    data = _as_dict(listing).get("data") or []
    return list(data)


def fetch_trace(trace_id: str, attempts: int = 8, delay: float = 8.0) -> Any:
    """
    Trace'ni observation'lari bilan birga o'qiydi.

    Langfuse ingestion asinxron: trace yozuvi darrov, observation'lar keyinroq
    paydo bo'ladi. Shuning uchun BO'SH javob ham "hali tayyor emas" deb
    hisoblanadi va qayta so'raladi (max ~attempts*delay soniya).
    """
    flush()
    client = get_client()
    last_error: Optional[Exception] = None
    last_trace: Any = None

    for i in range(attempts):
        try:
            trace = _get_trace_raw(client, trace_id)
            last_trace = trace
            if _trace_observations(trace):
                return trace
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        try:
            items = _list_observations(client, trace_id)
            if items:
                return {"id": trace_id, "observations": items, "_source": "observations.get_many"}
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        if i < attempts - 1:
            time.sleep(delay)

    if last_trace is not None:
        return last_trace
    raise ObservabilityError(f"Trace o'qilmadi ({trace_id}): {last_error}")


def summarize_trace(trace: Any) -> Dict[str, Any]:
    """Trace'dan observation nomlari va token hisoblarini chiqaradi."""
    data = _as_dict(trace)
    observations = data.get("observations") or []

    names: List[str] = []
    generations = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for obs in observations:
        od = _as_dict(obs)
        name = str(od.get("name") or "")
        if name:
            names.append(name)
        if str(od.get("type") or "").upper() == "GENERATION":
            generations += 1

        usage = od.get("usage_details") or od.get("usageDetails") or od.get("usage") or {}
        if not isinstance(usage, dict):
            usage = _as_dict(usage)

        def _num(*keys: str) -> int:
            for key in keys:
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    return int(value)
            return 0

        obs_in = _num("input", "promptTokens", "prompt_tokens", "input_tokens")
        obs_out = _num("output", "completionTokens", "completion_tokens", "output_tokens")
        obs_total = _num("total", "totalTokens", "total_tokens")

        input_tokens += obs_in
        output_tokens += obs_out
        total_tokens += obs_total or (obs_in + obs_out)

    # Zaxira: trace darajasidagi yig'indi
    if total_tokens == 0:
        trace_usage = data.get("usage_details") or data.get("usageDetails") or data.get("usage") or {}
        if not isinstance(trace_usage, dict):
            trace_usage = _as_dict(trace_usage)
        for key in ("total", "totalTokens", "total_tokens"):
            value = trace_usage.get(key)
            if isinstance(value, (int, float)):
                total_tokens = int(value)
                break

    return {
        "trace_id": data.get("id"),
        "observation_count": len(observations),
        "generation_count": generations,
        "names": names,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "source": data.get("_source", "trace.get"),
    }