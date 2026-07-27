"""
F13 check (backend qismi) - FastAPI SSE streaming.

Done when (backend): /api/stream jonli oqim beradi - agent qadamlari yakuniy
javobdan OLDIN va turli vaqtlarda keladi.

Avval boshqa terminalda serverni ishga tushiring:
    uvicorn backend.app:app --port 8000

Keyin:
    python scripts\\check_f13.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8000"
QUESTION = (
    "Engineering bo'limida nechta xodim bor va ularning o'rtacha oylik maoshi qancha? "
    "Shu o'rtacha maoshni Python bilan yillik summaga aylantirib ko'rsat."
)

results: list[tuple[str, bool, str]] = []


def record(label: str, ok: bool, detail: str = "") -> bool:
    results.append((label, ok, detail))
    print(("[OK]   " if ok else "[FAIL] ") + label + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 72)
    print("F13 - FastAPI backend + SSE streaming")
    print("=" * 72)

    try:
        health = httpx.get(
            f"{BASE}/api/health", timeout=10, trust_env=False
        ).json()
    except Exception as exc:  # noqa: BLE001
        record("1. /api/health", False, f"{type(exc).__name__}: {exc}")
        print("\nServer ishlamayapti. Boshqa terminalda ishga tushiring:")
        print("  uvicorn backend.app:app --port 8000")
        return 1

    record("1. /api/health", health.get("status") == "ok", json.dumps(health, ensure_ascii=False))
    record("2. Tracing yoqilgan (F12 davomi)", bool(health.get("tracing")))

    print("\n--- SSE oqimi ---")
    started = time.time()
    events: list[tuple[float, str, dict]] = []
    current_event = "message"

    try:
        with httpx.stream(
            "GET", f"{BASE}/api/stream", params={"question": QUESTION},
            timeout=300, trust_env=False,
        ) as response:
            record("3. /api/stream ulanishi", response.status_code == 200, f"HTTP {response.status_code}")
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue
                if line.startswith("data:"):
                    payload = json.loads(line.split(":", 1)[1].strip())
                    elapsed = round(time.time() - started, 2)
                    events.append((elapsed, current_event, payload))
                    node = payload.get("node") or payload.get("type")
                    print(f"  [{elapsed:6.2f}s] {current_event:<6} node={node}")
                    if current_event == "done":
                        break
    except Exception as exc:  # noqa: BLE001
        record("3. /api/stream ulanishi", False, f"{type(exc).__name__}: {exc}")
        return 1

    step_events = [e for e in events if e[1] == "step"]
    done_events = [e for e in events if e[1] == "done"]

    record("4. Kamida 3 ta qadam hodisasi keldi", len(step_events) >= 3, f"{len(step_events)} ta")

    # Haqiqiy streaming: qadamlar turli vaqtlarda kelishi kerak
    spread = (step_events[-1][0] - step_events[0][0]) if len(step_events) >= 2 else 0.0
    record(
        "5. Qadamlar vaqt bo'yicha tarqalgan (buferlanmagan)",
        spread > 0.5,
        f"birinchi va oxirgi qadam orasi {spread:.2f}s",
    )

    if done_events:
        done = done_events[-1][2]
        record("6. Yakuniy javob keldi", bool(done.get("answer")), f"{len(str(done.get('answer') or ''))} belgi")
        record("7. Langfuse trace havolasi qaytdi", bool(done.get("trace_url")), str(done.get("trace_url")))
        print(f"\n--- Javob ---\n{done.get('answer')}\n")
        # Qadamlar javobdan oldin kelganini isbotlaymiz
        record(
            "8. Qadamlar javobdan OLDIN kelgan",
            bool(step_events) and step_events[0][0] < done_events[-1][0],
            f"birinchi qadam {step_events[0][0]:.2f}s, javob {done_events[-1][0]:.2f}s",
        )
    else:
        record("6. Yakuniy javob keldi", False, "done hodisasi yo'q")

    failed = [label for label, ok, _ in results if not ok]
    print("\n" + "=" * 72)
    if failed:
        print(f"F13 (backend): FAIL -- {failed}")
        return 1
    print("F13 (backend): PASS")
    print("Keyingi qadam: frontend (Next.js) + Visual 2 screenshot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())