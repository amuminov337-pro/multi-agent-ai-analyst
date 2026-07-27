"""
F14 check - ommaviy havola ishlaydimi.

Done when: public URL real savolga javob beradi.

Ishga tushirish:
    python scripts\\check_f14.py --api https://<sizning-servis>.onrender.com
    python scripts\\check_f14.py --api https://... --web https://<vercel-domen>.vercel.app
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTION = (
    "Engineering bo'limida nechta xodim bor va ularning o'rtacha oylik maoshi qancha?"
)

results: list[tuple[str, bool, str]] = []


def record(label: str, ok: bool, detail: str = "") -> bool:
    results.append((label, ok, detail))
    print(("[OK]   " if ok else "[FAIL] ") + label + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, help="Backend public URL (Render)")
    parser.add_argument("--web", default="", help="Frontend public URL (Vercel), ixtiyoriy")
    args = parser.parse_args()

    api = args.api.rstrip("/")

    print("=" * 72)
    print("F14 - ommaviy havola")
    print(f"API: {api}")
    if args.web:
        print(f"WEB: {args.web}")
    print("=" * 72)

    # Render free uxlab qolgan bo'lsa, birinchi so'rov ~50-90 soniya kutadi
    print("\n--- /api/health (cold start uchun 120s kutiladi) ---")
    started = time.time()
    try:
        health = httpx.get(f"{api}/api/health", timeout=120, trust_env=False).json()
    except Exception as exc:  # noqa: BLE001
        record("1. Health javob berdi", False, f"{type(exc).__name__}: {exc}")
        return 1

    record(
        "1. Health javob berdi",
        health.get("status") == "ok",
        f"{json.dumps(health, ensure_ascii=False)} ({time.time() - started:.1f}s)",
    )
    record("2. HTTPS ishlatilgan", api.startswith("https://"), api)
    record("3. Tracing serverda yoqilgan", bool(health.get("tracing")))

    print("\n--- Ommaviy SSE oqimi ---")
    started = time.time()
    nodes: list[str] = []
    answer = ""
    trace_url = ""
    current_event = "message"

    try:
        with httpx.stream(
            "GET", f"{api}/api/stream", params={"question": QUESTION},
            timeout=300, trust_env=False,
        ) as response:
            record("4. /api/stream ulanishi", response.status_code == 200, f"HTTP {response.status_code}")
            for line in response.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue
                if line.startswith("data:"):
                    payload = json.loads(line.split(":", 1)[1].strip())
                    if current_event == "step" and payload.get("node"):
                        nodes.append(payload["node"])
                        print(f"  [{time.time() - started:6.2f}s] {payload['node']}")
                    if current_event == "done":
                        answer = payload.get("answer") or ""
                        trace_url = payload.get("trace_url") or ""
                        break
    except Exception as exc:  # noqa: BLE001
        record("4. /api/stream ulanishi", False, f"{type(exc).__name__}: {exc}")
        return 1

    record("5. Agent qadamlari keldi", len(nodes) >= 3, " -> ".join(nodes))
    record("6. Real javob qaytdi", len(answer.strip()) > 30, f"{len(answer)} belgi")
    record("7. Langfuse trace havolasi", bool(trace_url), trace_url)
    print(f"\n--- Javob ---\n{answer}\n")

    if args.web:
        try:
            page = httpx.get(args.web, timeout=60, trust_env=False)
            record("8. Frontend ochiladi", page.status_code == 200, f"HTTP {page.status_code}")
        except Exception as exc:  # noqa: BLE001
            record("8. Frontend ochiladi", False, f"{type(exc).__name__}: {exc}")

    failed = [label for label, ok, _ in results if not ok]
    print("=" * 72)
    if failed:
        print(f"F14: FAIL -- {failed}")
        return 1
    print("F14: PASS -- ommaviy havola real savolga javob berdi.")
    print("ESLATMA: submission uchun ikkala URL'ni README'ga yozing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())