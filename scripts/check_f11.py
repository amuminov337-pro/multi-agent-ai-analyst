"""F11 acceptance check — Evaluation harness (10 pts).

Done-when (from the guide): "running the harness prints a metrics table over a
test set of at least 10 questions."
Rubric also requires: RAGAS + LLM-judge, and a comparison WITH and WITHOUT the
critic.

Order matters. Run the harness first — it is the expensive, resumable part:

    python scripts/run_eval.py

Then run this check, which verifies the rubric conditions against the cached
results and costs almost nothing.

Five parts:
  1. QUESTION SET (no LLM): at least ten questions, each with ground truth,
     covering documents, SQL and code.
  2. DETERMINISTIC SCORING (no LLM): exact-match and judge parsing behave.
  3. RAGAS REACHABLE (no LLM): the library imports behind the shim and every
     metric we need is present.
  4. CACHED RESULTS: both modes were evaluated over the full set.
  5. TABLE AND REPORT: the comparison table has RAGAS rows, judge rows, and a
     column for each mode, and the report file exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.evaluation import (  # noqa: E402
    MODES,
    REPORT_PATH,
    EvalQuestion,
    comparison_table,
    exact_match,
    load_questions,
    load_record,
    parse_judge,
    summarise,
    write_report,
)
from ai.ragas_compat import import_ragas  # noqa: E402

MIN_QUESTIONS = 10


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def part_1_question_set():
    section("1) QUESTION SET")
    questions = load_questions()
    categories = {}
    for q in questions:
        categories[q.category] = categories.get(q.category, 0) + 1

    print(f"questions loaded : {len(questions)}")
    print(f"categories       : {categories}")
    missing_ref = [q.id for q in questions if not q.reference.strip()]
    missing_facts = [q.id for q in questions if not q.facts]
    duplicate_ids = len(questions) != len({q.id for q in questions})

    checks = [
        (f"at least {MIN_QUESTIONS} questions", len(questions) >= MIN_QUESTIONS),
        (f"every question has a reference (missing: {missing_ref or 'none'})",
         not missing_ref),
        (f"every question has checkable facts (missing: {missing_facts or 'none'})",
         not missing_facts),
        ("no duplicate ids", not duplicate_ids),
        ("document questions present", categories.get("doc", 0) >= 2),
        ("SQL questions present", categories.get("sql", 0) >= 2),
        ("code questions present", categories.get("code", 0) >= 1),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks), questions


def part_2_deterministic() -> bool:
    section("2) DETERMINISTIC SCORING — NO LLM")
    q = EvalQuestion(
        id="t", question="q", reference="Three customers.", facts=["3"], category="sql"
    )
    q_multi = EvalQuestion(
        id="t2", question="q", reference="r", facts=["3", "99.9"], category="mixed"
    )
    q_comma = EvalQuestion(
        id="t3", question="q", reference="r", facts=["2,000"], category="doc"
    )

    cases = [
        ("digit matches", exact_match(q, "There are 3 churned customers."), True),
        ("word matches digit", exact_match(q, "Three customers have churned."), True),
        ("wrong number rejected", exact_match(q, "Seven customers churned."), False),
        ("empty answer rejected", exact_match(q, ""), False),
        ("all facts required", exact_match(q_multi, "3 customers churned."), False),
        ("all facts present", exact_match(q_multi, "3 churned, 99.9% uptime."), True),
        ("comma-insensitive", exact_match(q_comma, "The budget is 2000 EUR."), True),
    ]
    ok_all = True
    for name, got, expected in cases:
        ok = got is expected
        ok_all = ok_all and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: got {got}")

    print("\n  judge parsing:")
    parse_cases = [
        ("clean format", "SCORE: 5\nREASON: Correct.", 5),
        ("lowercase", "score: 2\nreason: wrong number", 2),
        ("bare digit", "4", 4),
        ("noisy preamble", "Here is my grade.\nSCORE: 1\nREASON: no answer", 1),
    ]
    for name, text, expected in parse_cases:
        score, _ = parse_judge(text)
        ok = score == expected
        ok_all = ok_all and ok
        print(f"    [{'OK' if ok else 'FAIL'}] {name}: got {score}")

    return ok_all


def part_3_ragas_reachable() -> bool:
    section("3) RAGAS REACHABLE BEHIND THE SHIM — NO LLM")
    try:
        bundle = import_ragas()
    except Exception as exc:
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        return False

    print(f"  version        : {bundle.version}")
    print(f"  RunConfig      : {'available' if bundle.RunConfig else 'not available'}")
    checks = [
        ("evaluate", bundle.evaluate is not None),
        ("EvaluationDataset", bundle.EvaluationDataset is not None),
        ("SingleTurnSample", bundle.SingleTurnSample is not None),
        ("faithfulness", bundle.faithfulness is not None),
        ("answer_relevancy", bundle.answer_relevancy is not None),
        ("context_precision", bundle.context_precision is not None),
        ("context_recall", bundle.context_recall is not None),
        ("LangchainLLMWrapper", bundle.LangchainLLMWrapper is not None),
        ("LangchainEmbeddingsWrapper", bundle.LangchainEmbeddingsWrapper is not None),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_4_cached_results(questions):
    section("4) CACHED RESULTS FROM THE HARNESS")
    results = {}
    ok_all = True

    for mode in MODES:
        records = []
        missing = []
        for q in questions:
            record = load_record(mode, q.id)
            if record and record.get("answer"):
                records.append(record)
            else:
                missing.append(q.id)
        results[mode] = records

        judged = sum(1 for r in records if isinstance(r.get("judge_score"), int))
        matched = sum(1 for r in records if r.get("exact_match"))
        print(f"\n  {mode}:")
        print(f"    records      : {len(records)}/{len(questions)}")
        print(f"    judged       : {judged}")
        print(f"    exact matches: {matched}")
        if missing:
            print(f"    MISSING      : {missing}")

        checks = [
            (f"{mode}: all questions evaluated", not missing),
            (f"{mode}: all records judged", judged == len(records) and records != []),
        ]
        for name, ok in checks:
            ok_all = ok_all and ok
            print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if not ok_all:
        print("\n  -> run `python scripts/run_eval.py` first; it is resumable.")
    return ok_all, results


def part_5_table(results) -> bool:
    section("5) COMPARISON TABLE AND REPORT")
    ragas_cache = {}
    for mode in MODES:
        path = ROOT / "data" / "eval_results" / mode / "_ragas.json"
        if path.exists():
            import json

            try:
                ragas_cache[mode] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                ragas_cache[mode] = {}

    summaries = {
        mode: summarise(records, ragas_cache.get(mode))
        for mode, records in results.items()
        if records
    }
    table = comparison_table(summaries)
    print()
    print(table)

    report = write_report(results, summaries)
    print(f"\nreport: {report}")

    ragas_rows = [line for line in table.splitlines() if line.startswith("| RAGAS ")]
    print(f"\nRAGAS rows in the table: {len(ragas_rows)}")
    for row in ragas_rows:
        print(f"  {row}")

    checks = [
        ("both modes are in the summary", len(summaries) == len(MODES)),
        ("table has a with-critic column", "With critic" in table),
        ("table has a without-critic column", "Without critic" in table),
        ("table has a judge row", "LLM judge mean" in table),
        ("table has RAGAS metric rows", len(ragas_rows) >= 3),
        ("table has an exact-match row", "Exact match rate" in table),
        ("report file written", REPORT_PATH.exists()),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def main() -> int:
    ok_questions, questions = part_1_question_set()
    ok_deterministic = part_2_deterministic()
    ok_ragas = part_3_ragas_reachable()
    ok_cached, results = part_4_cached_results(questions)
    ok_table = part_5_table(results) if any(results.values()) else False

    results_map = {
        "question set valid": ok_questions,
        "deterministic scoring": ok_deterministic,
        "RAGAS reachable": ok_ragas,
        "both modes evaluated": ok_cached,
        "comparison table produced": ok_table,
    }

    section("RESULT")
    for name, ok in results_map.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if all(results_map.values()):
        print("\nPASS — F11 done (10/10)")
        print(f"  - {len(questions)} questions, each with verifiable ground truth")
        print("  - RAGAS metrics + LLM judge + deterministic exact match")
        print("  - with-critic vs without-critic comparison table produced")
        print(f"  - Visual 4 saved to {REPORT_PATH.relative_to(ROOT)}")
        return 0
    print("\nFAIL — F11 not complete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())