"""Run the F11 evaluation harness and write the report.

    python scripts/run_eval.py                  # full run, both modes, RAGAS on
    python scripts/run_eval.py --limit 2        # quick smoke run
    python scripts/run_eval.py --skip-ragas     # judge only (cheaper, faster)
    python scripts/run_eval.py --force          # ignore the cache, redo everything

Everything is cached per question per mode under data/eval_results/, so an
interrupted run — a quota limit, a network drop, Ctrl+C — resumes where it
stopped instead of starting over.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.evaluation import (  # noqa: E402
    MODES,
    comparison_table,
    evaluate_all,
    load_questions,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="F11 evaluation harness.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions.")
    parser.add_argument("--mode", choices=["both", "with", "without"], default="both",
                        help="Which critic mode(s) to evaluate.")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Skip RAGAS metrics (much cheaper).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute everything, ignoring the cache.")
    parser.add_argument("--use-proxy", action="store_true",
                        help="Use the class proxy for judge+RAGAS calls (quota relief).")
    args = parser.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    modes = MODES
    if args.mode == "with":
        modes = ("with_critic",)
    elif args.mode == "without":
        modes = ("without_critic",)

    print("=" * 64)
    print("F11 EVALUATION HARNESS")
    print("=" * 64)
    print(f"questions : {len(questions)}")
    print(f"modes     : {', '.join(modes)}")
    print(f"ragas     : {'skipped' if args.skip_ragas else 'enabled'}")
    print(f"cache     : {'ignored (--force)' if args.force else 'used where available'}")
    print(f"proxy     : {'yes' if args.use_proxy else 'no'}")

    results, summaries = evaluate_all(
        questions=questions,
        modes=modes,
        force=args.force,
        skip_ragas=args.skip_ragas,
        use_proxy=args.use_proxy,
    )

    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print()
    print(comparison_table(summaries))

    report = write_report(results, summaries)
    print(f"\nreport written: {report}")
    print("  -> this table is Visual 4 of the final submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())