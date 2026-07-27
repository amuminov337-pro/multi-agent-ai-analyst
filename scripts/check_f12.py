"""
F12 check - Langfuse observability.

Done when: Langfuse trace'da to'liq yo'l (supervisor -> data -> code -> critic)
va token hisoblari ko'rinadi.

Ishga tushirish (repo ildizidan):
    python scripts\\check_f12.py                      # to'liq run + tekshiruv (kvota sarflanadi)
    python scripts\\check_f12.py --trace-id <ID>      # mavjud trace'ni tekshirish (BEPUL)
"""

from __future__ import annotations

import pathlib
import sys
import traceback
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai import observability as obs  # noqa: E402

QUESTION = (
    "Engineering bo'limida nechta xodim bor va ularning o'rtacha oylik maoshi qancha? "
    "Shu o'rtacha maoshni Python bilan hisoblab, yillik summasini va 12 oyga "
    "taqsimlangan jadvalini chiqar."
)

REQUIRED_STEPS = ["supervisor", "data", "code", "critic"]

results: list[tuple[str, bool, str]] = []


def record(label: str, ok: bool, detail: str = "") -> bool:
    results.append((label, ok, detail))
    mark = "[OK]  " if ok else "[FAIL]"
    print(f"{mark} {label}" + (f" -- {detail}" if detail else ""))
    return ok


def parse_trace_id(argv: list[str]) -> str | None:
    if "--trace-id" in argv:
        index = argv.index("--trace-id")
        if index + 1 < len(argv):
            return argv[index + 1]
    return None


def main() -> int:
    existing_trace_id = parse_trace_id(sys.argv[1:])

    print("=" * 72)
    print("F12 - Langfuse observability")
    if existing_trace_id:
        print(f"REJIM: mavjud trace tekshiriladi ({existing_trace_id}) - graf yugurtirilmaydi")
    print("=" * 72)

    sdk = obs.load_sdk()
    record(
        "1. Langfuse SDK import",
        sdk["handler_cls"] is not None,
        f"version={obs.sdk_version()} major={sdk['major']} err={sdk['import_error']}",
    )
    if sdk["handler_cls"] is None:
        return 1

    creds = obs.credentials()
    has_keys = bool(creds["public_key"] and creds["secret_key"])
    record(
        "2. .env kalitlari",
        has_keys,
        f"host={creds['host']} public_key={'set' if creds['public_key'] else 'MISSING'}",
    )
    if not has_keys:
        return 1

    try:
        ok_auth = obs.auth_check()
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc())
        record("3. Langfuse auth_check", False, f"{type(exc).__name__}: {exc}")
        return 1
    record("3. Langfuse auth_check", ok_auth)
    if not ok_auth:
        return 1

    trace_id = existing_trace_id
    session_id = ""

    if not existing_trace_id:
        session_id = f"f12-check-{uuid.uuid4().hex[:8]}"
        print("\n--- Grafni tracing bilan ishga tushiramiz (bir necha model chaqiruvi) ---")
        print(f"Savol: {QUESTION}\n")
        try:
            run = obs.run_traced(
                QUESTION,
                trace_name="f12-check-full-path",
                session_id=session_id,
                tags=["capstone", "F12", "check"],
            )
        except Exception as exc:  # noqa: BLE001
            print(traceback.format_exc())
            record("4. Tracing bilan run", False, f"{type(exc).__name__}: {exc}")
            return 1

        state = run["state"]
        trace_id = run["trace_id"]
        record("4. Tracing bilan run", bool(run["traced"]), f"trace_id={trace_id}")

        print("\n--- Bosqichlar (state['steps']) ---")
        for step in state.get("steps", []):
            print(f"  - {step}")
        print(f"\n  visited   : {state.get('visited')}")
        print(f"  critic_ok : {state.get('critic_ok')}")
        print(f"  revisions : {state.get('revisions')}")
        print(f"\n--- Javob ---\n{state.get('answer')}\n")

        local_blob = " ".join(state.get("steps", [])).lower() + " " + " ".join(
            str(v) for v in (state.get("visited") or [])
        ).lower()
        missing_local = [s for s in REQUIRED_STEPS if s not in local_blob]
        record(
            "5. Lokal yo'l: supervisor -> data -> code -> critic",
            not missing_local,
            f"yetishmaydi: {missing_local}" if missing_local else "",
        )

    if not trace_id:
        record("6. Trace id", False, "Handler trace id qaytarmadi")
        return 1

    print("\n--- Trace Langfuse'da indekslanishini kutamiz (max ~60s) ---")
    try:
        trace = obs.fetch_trace(trace_id)
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc())
        record("6. Trace serverdan o'qildi", False, f"{type(exc).__name__}: {exc}")
        return 1

    summary = obs.summarize_trace(trace)
    record(
        "6. Trace serverdan o'qildi",
        summary["observation_count"] > 0,
        f"{summary['observation_count']} observation (manba: {summary['source']})",
    )

    print("\n--- Trace observation'lari ---")
    for name in summary["names"]:
        print(f"  - {name}")

    remote_blob = " ".join(summary["names"]).lower()
    missing_remote = [s for s in REQUIRED_STEPS if s not in remote_blob]
    record(
        "7. Trace'da to'liq yo'l ko'rinadi",
        not missing_remote,
        f"yetishmaydi: {missing_remote}" if missing_remote else "",
    )

    print(
        f"\n  input tokens : {summary['input_tokens']}"
        f"\n  output tokens: {summary['output_tokens']}"
        f"\n  total tokens : {summary['total_tokens']}"
        f"\n  generations  : {summary['generation_count']}"
    )
    record(
        "8. Token hisoblari mavjud",
        summary["total_tokens"] > 0 and summary["generation_count"] > 0,
        f"total={summary['total_tokens']}",
    )

    print("\n" + "=" * 72)
    print(f"TRACE URL: {obs.trace_url(trace_id)}")
    if session_id:
        print(f"SESSION  : {session_id}")
    print("=" * 72)

    failed = [label for label, ok, _ in results if not ok]
    if failed:
        print(f"\nF12: FAIL -- {len(failed)} shart bajarilmadi: {failed}")
        return 1

    print("\nF12: PASS -- barcha shartlar bajarildi.")
    print("ESLATMA (Visual 3): TRACE URL'ni brauzerda oching, daraxtni yoying va")
    print("to'liq yo'l + token badge ko'rinib turgan screenshot'ni saqlang:")
    print("  documents/langfuse_trace.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())