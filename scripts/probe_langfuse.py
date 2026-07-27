"""
F12 diagnostikasi - Langfuse trace'ida observation va token bormi.

Model chaqirmaydi, faqat Langfuse API'dan o'qiydi (arzon, xohlagancha qayta ishlatsa bo'ladi).

Ishga tushirish (repo ildizidan):
    python scripts\\probe_langfuse.py 891eecaef1b736e4fc8eff41b73932df
    python scripts\\probe_langfuse.py            # oxirgi ma'lum trace id
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai import observability as obs  # noqa: E402

DEFAULT_TRACE_ID = "891eecaef1b736e4fc8eff41b73932df"
WAITS = [0, 5, 10, 15, 20, 30, 30]  # jami ~110 soniya


def as_dict(obj):
    return obs._as_dict(obj)  # bir xil tolerant konvertor


def show_observations(items) -> int:
    count = 0
    for item in items or []:
        d = as_dict(item)
        usage = d.get("usage_details") or d.get("usageDetails") or d.get("usage") or {}
        print(
            f"    - name={d.get('name')!r} "
            f"type={d.get('type')!r} "
            f"model={d.get('model')!r} "
            f"usage={usage}"
        )
        count += 1
    return count


def main() -> int:
    trace_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRACE_ID
    print("=" * 72)
    print(f"Langfuse probe - trace_id={trace_id}")
    print(f"SDK version: {obs.sdk_version()}")
    print("=" * 72)

    try:
        client = obs.get_client()
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc())
        print(f"KLIENT XATOSI: {type(exc).__name__}: {exc}")
        return 1

    print(f"client type      : {type(client)}")
    print(f"client.api bor?  : {hasattr(client, 'api')}")
    api = getattr(client, "api", None)
    print(f"api.trace bor?   : {hasattr(api, 'trace') if api else False}")
    print(f"api.observations : {hasattr(api, 'observations') if api else False}")

    # Buferda qolgan narsa bo'lsa yuboramiz
    try:
        client.flush()
        print("flush()          : OK")
    except Exception as exc:  # noqa: BLE001
        print(f"flush()          : XATO {type(exc).__name__}: {exc}")

    found = 0
    elapsed = 0

    for wait in WAITS:
        if wait:
            print(f"\n--- {wait}s kutamiz (jami {elapsed + wait}s) ---")
            time.sleep(wait)
        elapsed += wait

        # 1-yo'l: trace.get
        try:
            trace = api.trace.get(trace_id)
            d = as_dict(trace)
            observations = d.get("observations") or []
            print(f"[trace.get] kalitlar: {sorted(d.keys())}")
            print(f"[trace.get] name={d.get('name')!r} observations={len(observations)}")
            found = show_observations(observations)
        except Exception as exc:  # noqa: BLE001
            print(f"[trace.get] XATO: {type(exc).__name__}: {exc}")

        # 2-yo'l: observations ro'yxatini alohida so'raymiz
        try:
            listing = api.observations.get_many(trace_id=trace_id)
            ld = as_dict(listing)
            items = ld.get("data") or []
            print(f"[observations.get_many] soni={len(items)}")
            found = max(found, show_observations(items))
        except Exception as exc:  # noqa: BLE001
            print(f"[observations.get_many] XATO: {type(exc).__name__}: {exc}")

        if found:
            print(f"\nTOPILDI: {found} ta observation, {elapsed}s dan keyin.")
            print("=> Sabab A: ingestion kechikishi. fetch_trace() ni tuzatamiz.")
            return 0

    print("\nHECH NARSA TOPILMADI (~110s ichida).")
    print("=> Sabab B: span'lar yuborilmagan. Handler ulanishini qayta ko'ramiz.")
    print(f"Brauzerda tekshiring: {obs.trace_url(trace_id)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())