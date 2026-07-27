"""Evaluation harness: RAGAS + LLM-judge over a fixed question set (F11).

The rubric requires a metrics table comparing the system WITH and WITHOUT the
critic over at least ten questions. That is what this module produces.

Structure, and why it is split in two phases:

  PHASE 1 — run_phase(): answer every question in both modes and CACHE each
    result to disk as its own JSON file.
  PHASE 2 — score_phase(): read the cache and score it (deterministic exact
    match, LLM judge, then RAGAS).

Phase 1 costs roughly 150 model calls and phase 2 another 200. A daily quota
can run out in the middle of that, so every unit of work is cached the moment
it completes and skipped on the next run. Losing an afternoon of quota must
never mean starting over.

Three metric families, measuring genuinely different things:

  * EXACT MATCH (free, deterministic) — does the answer contain the
    ground-truth fact? No model involved, so it cannot flatter the system.
  * LLM JUDGE (1-5, against the reference) — is the answer actually CORRECT?
  * RAGAS (faithfulness, answer relevancy, context precision/recall) — is the
    answer GROUNDED in the evidence, and was the right evidence gathered?

Judge and RAGAS do not overlap: an answer built on wrong evidence can score
high on faithfulness (it faithfully reflects that evidence) while the judge
correctly marks it wrong.

Evaluation always runs with use_memory=False. With memory on, a re-run would
recall its own previous answer and the comparison would stop being
reproducible.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai.config import DATA_DIR, PROJECT_ROOT
from ai.graph import run
from ai.llm import ask
from ai.ragas_compat import import_ragas

QUESTIONS_PATH = PROJECT_ROOT / "documents" / "eval_questions.json"
RESULTS_DIR = DATA_DIR / "eval_results"
REPORT_PATH = PROJECT_ROOT / "documents" / "eval_results.md"

MODES: Tuple[str, ...] = ("with_critic", "without_critic")
MODE_LABELS = {"with_critic": "With critic", "without_critic": "Without critic"}

#: Digit -> English word, so "Three customers" counts as containing "3".
NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
}

JUDGE_PROMPT = """You are grading an AI system's answer against a reference answer.

Question: {question}

Reference answer (ground truth): {reference}

System answer: {answer}

Score the system answer from 1 to 5:
5 = fully correct and complete; every fact matches the reference
4 = correct, but missing a minor detail the question asked for
3 = partially correct; one required fact is right and another is wrong or absent
2 = mostly wrong, though it addresses the right topic
1 = wrong, or it fails to answer the question at all

Grade on CORRECTNESS only. Ignore wording, length and formatting. A number
written as a word ("three") is the same as the digit ("3").

Reply with exactly two lines:
SCORE: <a single digit 1-5>
REASON: <one short sentence>"""


# --------------------------------------------------------------------------
# question set
# --------------------------------------------------------------------------


@dataclass
class EvalQuestion:
    """One question plus its verifiable ground truth."""

    id: str
    question: str
    reference: str
    facts: List[str]
    category: str = "doc"
    expected_agent: str = ""


def load_questions(path: Optional[Path] = None) -> List[EvalQuestion]:
    """Read the fixed evaluation set from disk."""
    source = path or QUESTIONS_PATH
    if not source.exists():
        raise FileNotFoundError(f"Evaluation set not found at {source}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    items = payload["questions"] if isinstance(payload, dict) else payload

    questions = [
        EvalQuestion(
            id=item["id"],
            question=item["question"],
            reference=item["reference"],
            facts=list(item.get("facts") or []),
            category=item.get("category", "doc"),
            expected_agent=item.get("expected_agent", ""),
        )
        for item in items
    ]
    if not questions:
        raise ValueError(f"No questions found in {source}")
    return questions


# --------------------------------------------------------------------------
# deterministic scoring
# --------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase and strip thousands separators so 2,000 matches 2000."""
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", (text or "").lower())


def fact_present(fact: str, answer: str) -> bool:
    """Is this ground-truth fact in the answer, as a digit or as a word?"""
    haystack = _normalise(answer)
    needle = _normalise(fact)
    if not needle:
        return False
    if needle in haystack:
        return True
    word = NUMBER_WORDS.get(needle)
    return bool(word and re.search(rf"\b{word}\b", haystack))


def exact_match(question: EvalQuestion, answer: str) -> bool:
    """True when every required fact appears in the answer.

    Deliberately model-free: this is the one metric that cannot be talked into
    a better score, which makes it the anchor for the whole comparison.
    """
    if not question.facts:
        return False
    return all(fact_present(fact, answer) for fact in question.facts)


# --------------------------------------------------------------------------
# LLM judge
# --------------------------------------------------------------------------


def parse_judge(text: str) -> Tuple[Optional[int], str]:
    """Pull (score, reason) out of the judge's reply. Tolerant of stray text."""
    score: Optional[int] = None
    reason = ""

    match = re.search(r"score\s*[:=]?\s*([1-5])", text or "", re.IGNORECASE)
    if match:
        score = int(match.group(1))
    else:
        loose = re.search(r"\b([1-5])\b", text or "")
        if loose:
            score = int(loose.group(1))

    reason_match = re.search(r"reason\s*[:=]?\s*(.+)", text or "", re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip().splitlines()[0].strip()
    elif text:
        reason = text.strip().splitlines()[-1].strip()

    return score, reason[:300]


def judge_answer(
    question: EvalQuestion, answer: str, use_proxy: bool = False
) -> Tuple[Optional[int], str]:
    """Score one answer 1-5 against the reference. Never raises."""
    if not (answer or "").strip():
        return 1, "No answer was produced."

    prompt = JUDGE_PROMPT.format(
        question=question.question,
        reference=question.reference,
        answer=answer,
    )
    try:
        if use_proxy:
            from ai.llm_proxy import ask_proxy
            return parse_judge(ask_proxy(prompt))
        return parse_judge(ask(prompt))
    except Exception as exc:
        return None, f"judge unavailable ({type(exc).__name__})"


# --------------------------------------------------------------------------
# phase 1: run the system, cache every result
# --------------------------------------------------------------------------


def _record_path(mode: str, question_id: str) -> Path:
    return RESULTS_DIR / mode / f"{question_id}.json"


def load_record(mode: str, question_id: str) -> Optional[dict]:
    """Read a cached result, or None if it is absent or unreadable."""
    path = _record_path(mode, question_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_record(record: dict) -> None:
    """Persist one result immediately, so quota exhaustion loses nothing."""
    path = _record_path(record["mode"], record["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def build_contexts(final: dict) -> List[str]:
    """Flatten the gathered evidence into RAGAS "retrieved contexts".

    In a multi-agent system the retrieved context is not only document chunks:
    a SQL result or a computed value is evidence in exactly the same sense, and
    excluding them would make faithfulness meaningless for numeric questions.
    RAGAS rejects an empty list, so a placeholder stands in when nothing was
    gathered — that case is itself a finding worth measuring.
    """
    contexts: List[str] = [c for c in (final.get("documents") or []) if (c or "").strip()]
    for key in ("sql_result", "code_result"):
        value = final.get(key)
        if value and str(value).strip():
            contexts.append(str(value))
    return contexts or ["(no evidence was gathered)"]


def run_one(question: EvalQuestion, mode: str) -> dict:
    """Answer one question in one mode and return its record."""
    use_critic = mode == "with_critic"
    started = time.perf_counter()
    final = run(question.question, use_critic=use_critic, use_memory=False)
    duration = time.perf_counter() - started

    answer = (final.get("answer") or "").strip()
    return {
        "id": question.id,
        "mode": mode,
        "category": question.category,
        "question": question.question,
        "reference": question.reference,
        "facts": question.facts,
        "expected_agent": question.expected_agent,
        "answer": answer,
        "contexts": build_contexts(final),
        "steps": list(final.get("steps") or []),
        "visited": list(final.get("visited") or []),
        "critic_ok": bool(final.get("critic_ok")),
        "critic_reason": final.get("critic_reason") or "",
        "revisions": int(final.get("revisions", 0)),
        "duration_seconds": round(duration, 2),
        "exact_match": exact_match(question, answer),
    }


def run_phase(
    questions: List[EvalQuestion],
    modes: Tuple[str, ...] = MODES,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, List[dict]]:
    """Phase 1: produce (or reuse) one record per question per mode."""
    results: Dict[str, List[dict]] = {mode: [] for mode in modes}

    for mode in modes:
        for question in questions:
            cached = None if force else load_record(mode, question.id)
            if cached and cached.get("answer"):
                if verbose:
                    print(f"  [cached] {mode:<15} {question.id}")
                results[mode].append(cached)
                continue

            if verbose:
                print(f"  [run   ] {mode:<15} {question.id} …", end="", flush=True)
            record = run_one(question, mode)
            save_record(record)
            results[mode].append(record)
            if verbose:
                mark = "OK " if record["exact_match"] else "MISS"
                print(f" {mark} ({record['duration_seconds']}s)")

    return results


# --------------------------------------------------------------------------
# phase 2: judge + RAGAS
# --------------------------------------------------------------------------


def judge_phase(
    results: Dict[str, List[dict]],
    force: bool = False,
    verbose: bool = True,
    use_proxy: bool = False,
) -> None:
    """Add judge scores to every record, caching as it goes."""
    for mode, records in results.items():
        for record in records:
            if not force and record.get("judge_score") is not None:
                if verbose:
                    print(f"  [cached] judge {mode:<15} {record['id']}")
                continue

            question = EvalQuestion(
                id=record["id"],
                question=record["question"],
                reference=record["reference"],
                facts=record.get("facts") or [],
            )
            score, reason = judge_answer(
                question, record.get("answer") or "", use_proxy=use_proxy
            )
            record["judge_score"] = score
            record["judge_reason"] = reason
            save_record(record)
            if verbose:
                print(f"  [judge ] {mode:<15} {record['id']} -> {score}")


def _ragas_path(mode: str) -> Path:
    return RESULTS_DIR / mode / "_ragas.json"


def ragas_phase(
    results: Dict[str, List[dict]],
    force: bool = False,
    verbose: bool = True,
    max_workers: int = 1,
    use_proxy: bool = False,
) -> Dict[str, dict]:
    """Score every mode with real RAGAS metrics.

    Returns {mode: {metric: mean_score}}. A mode that fails records its error
    instead of raising, so one broken metric cannot destroy the whole report.
    """
    if use_proxy:
        from ai.llm_proxy import get_proxy_llm, get_proxy_embeddings
        raw_llm = get_proxy_llm("gemini-flash-lite")
        raw_embeddings = get_proxy_embeddings("gemini-embedding")
    else:
        from ai.llm import get_llm
        from ai.vectorstore import get_embeddings
        raw_llm = get_llm()
        raw_embeddings = get_embeddings()

    bundle = import_ragas()
    if verbose:
        print(f"  RAGAS version: {bundle.version} (proxy={'yes' if use_proxy else 'no'})")

    wrapped_llm = bundle.LangchainLLMWrapper(raw_llm)
    wrapped_embeddings = bundle.LangchainEmbeddingsWrapper(raw_embeddings)
    # answer_relevancy asks the LLM for `strictness` candidate questions in ONE
    # call (n=strictness). Gemini rejects n>1 with "Multiple candidates is not
    # enabled for this model", which is why the metric returned nan. Dropping
    # to 1 keeps the metric meaningful while staying inside what Gemini allows.
    try:
        bundle.answer_relevancy.strictness = 1
    except Exception:
        pass

    metrics = [
        bundle.faithfulness,
        bundle.answer_relevancy,
        bundle.context_precision,
        bundle.context_recall,
    ]

    # Cap parallelism: the free Gemini tier limits requests per minute, and
    # RAGAS fans out aggressively by default.
    kwargs: Dict[str, Any] = {}
    if bundle.RunConfig is not None:
        kwargs["run_config"] = bundle.RunConfig(max_workers=max_workers)

    scores: Dict[str, dict] = {}
    for mode, records in results.items():
        cache = _ragas_path(mode)
        if not force and cache.exists():
            try:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                # The cache is only valid for the sample count it was computed
                # over: a 2-question smoke run must never be reused as the
                # score for a 12-question evaluation.
                if cached.get("_sample_count") == len(records):
                    scores[mode] = cached
                    if verbose:
                        print(f"  [cached] ragas {mode} ({len(records)} samples)")
                    continue
                if verbose:
                    print(
                        f"  [stale ] ragas {mode}: cached for "
                        f"{cached.get('_sample_count')} samples, now {len(records)} "
                        "— recomputing"
                    )
            except Exception:
                pass

        samples = [
            bundle.SingleTurnSample(
                user_input=record["question"],
                response=record.get("answer") or "",
                retrieved_contexts=list(record.get("contexts") or ["(none)"]),
                reference=record["reference"],
            )
            for record in records
        ]
        dataset = bundle.EvaluationDataset(samples=samples)

        if verbose:
            print(f"  [ragas ] {mode}: scoring {len(samples)} samples …")
        try:
            result = bundle.evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=wrapped_llm,
                embeddings=wrapped_embeddings,
                **kwargs,
            )
            mode_scores = _extract_ragas_scores(result)
            metric_names = [getattr(m, "name", None) for m in metrics]
            metric_names = [n for n in metric_names if n]
            validity = _sample_validity(result, metric_names)
            if validity:
                mode_scores["_valid_samples"] = validity
        except Exception as exc:
            mode_scores = {"error": f"{type(exc).__name__}: {exc}"}

        mode_scores["_sample_count"] = len(records)
        scores[mode] = mode_scores
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(mode_scores, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return scores


def _extract_ragas_scores(result: Any) -> Dict[str, float]:
    """Read mean per-metric scores out of a RAGAS result object.

    The result type has changed shape across RAGAS versions, so several access
    patterns are tried before giving up.
    """
    # Newer versions expose _repr_dict / to_pandas.
    for attr in ("_repr_dict", "scores_dict"):
        candidate = getattr(result, attr, None)
        if isinstance(candidate, dict) and candidate:
            return {
                key: round(float(value), 4)
                for key, value in candidate.items()
                if isinstance(value, (int, float))
            }

    try:
        frame = result.to_pandas()
        numeric = frame.select_dtypes("number")
        return {col: round(float(numeric[col].mean()), 4) for col in numeric.columns}
    except Exception:
        pass

    if isinstance(result, dict):
        return {
            key: round(float(value), 4)
            for key, value in result.items()
            if isinstance(value, (int, float))
        }

    return {"error": f"could not read scores from {type(result).__name__}"}


def _sample_validity(result: Any, metric_names: List[str]) -> Dict[str, Dict[str, int]]:
    """Count, per metric, how many underlying per-sample scores are real
    numbers rather than NaN.

    RAGAS silently turns a failed per-sample judgment (rate limit, timeout,
    malformed output) into NaN and averages over whatever is left. A mean
    built from 7 of 12 samples looks identical to one built from all 12
    unless this is checked explicitly — which is exactly what a rate-limited
    run can produce.
    """
    try:
        frame = result.to_pandas()
    except Exception:
        return {}

    counts: Dict[str, Dict[str, int]] = {}
    total = len(frame)
    for name in metric_names:
        if name in frame.columns:
            valid = int(frame[name].notna().sum())
            counts[name] = {"valid": valid, "total": total}
    return counts


# --------------------------------------------------------------------------
# aggregation and reporting
# --------------------------------------------------------------------------


@dataclass
class ModeSummary:
    """Aggregated numbers for one mode."""

    mode: str
    count: int
    exact_match_rate: float
    judge_mean: Optional[float]
    judge_scored: int
    verified_rate: Optional[float]
    revisions_mean: float
    duration_mean: float
    ragas: Dict[str, Any] = field(default_factory=dict)


def summarise(
    records: List[dict],
    ragas_scores: Optional[dict] = None,
) -> ModeSummary:
    """Collapse one mode's records into headline numbers."""
    mode = records[0]["mode"] if records else "unknown"
    count = len(records)
    matches = sum(1 for r in records if r.get("exact_match"))
    judged = [r["judge_score"] for r in records if isinstance(r.get("judge_score"), int)]
    verified = [bool(r.get("critic_ok")) for r in records]

    return ModeSummary(
        mode=mode,
        count=count,
        exact_match_rate=round(matches / count, 4) if count else 0.0,
        judge_mean=round(statistics.fmean(judged), 3) if judged else None,
        judge_scored=len(judged),
        verified_rate=(
            round(sum(verified) / count, 4) if count and mode == "with_critic" else None
        ),
        revisions_mean=round(
            statistics.fmean([r.get("revisions", 0) for r in records]), 3
        ) if count else 0.0,
        duration_mean=round(
            statistics.fmean([r.get("duration_seconds", 0.0) for r in records]), 2
        ) if count else 0.0,
        ragas=dict(ragas_scores or {}),
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _delta(with_value: Any, without_value: Any) -> str:
    if not isinstance(with_value, (int, float)) or not isinstance(
        without_value, (int, float)
    ):
        return "n/a"
    diff = float(with_value) - float(without_value)
    return f"{diff:+.3f}"


def comparison_table(summaries: Dict[str, ModeSummary]) -> str:
    """Markdown table: with critic vs without critic. This is Visual 4."""
    with_c = summaries.get("with_critic")
    without_c = summaries.get("without_critic")

    rows: List[Tuple[str, Any, Any]] = [
        ("Questions evaluated", with_c.count if with_c else None,
         without_c.count if without_c else None),
        ("Exact match rate", with_c.exact_match_rate if with_c else None,
         without_c.exact_match_rate if without_c else None),
        ("LLM judge mean (1-5)", with_c.judge_mean if with_c else None,
         without_c.judge_mean if without_c else None),
    ]

    metric_names: List[str] = []
    for summary in (with_c, without_c):
        for key in (summary.ragas if summary else {}):
            if key not in ("error", "_sample_count", "_valid_samples") and key not in metric_names:
                metric_names.append(key)
    for name in metric_names:
        rows.append((
            f"RAGAS {name}",
            (with_c.ragas.get(name) if with_c else None),
            (without_c.ragas.get(name) if without_c else None),
        ))

    rows.extend([
        ("Mean revisions", with_c.revisions_mean if with_c else None,
         without_c.revisions_mean if without_c else None),
        ("Mean seconds per question", with_c.duration_mean if with_c else None,
         without_c.duration_mean if without_c else None),
        ("Critic-verified rate", with_c.verified_rate if with_c else None, None),
    ])

    lines = [
        "| Metric | With critic | Without critic | Delta |",
        "| --- | --- | --- | --- |",
    ]
    for label, a, b in rows:
        lines.append(f"| {label} | {_fmt(a)} | {_fmt(b)} | {_delta(a, b)} |")

    errors = [
        f"- `{mode}`: {summary.ragas['error']}"
        for mode, summary in summaries.items()
        if summary and summary.ragas.get("error")
    ]
    if errors:
        lines.append("")
        lines.append("RAGAS errors:")
        lines.extend(errors)

    incomplete = []
    for mode, summary in summaries.items():
        if not summary:
            continue
        for metric, counts in (summary.ragas.get("_valid_samples") or {}).items():
            if counts.get("valid", 0) < counts.get("total", 0):
                incomplete.append(
                    f"- `{mode}` / RAGAS {metric}: only {counts['valid']}/"
                    f"{counts['total']} samples were actually scored (the rest "
                    "failed — likely rate limiting — and were silently dropped "
                    "from the mean by RAGAS)."
                )
    if incomplete:
        lines.append("")
        lines.append("**Data quality warning — incomplete RAGAS samples:**")
        lines.extend(incomplete)

    return "\n".join(lines)


def per_question_table(results: Dict[str, List[dict]]) -> str:
    """Per-question breakdown — the raw material for the error analysis."""
    by_id: Dict[str, Dict[str, dict]] = {}
    for mode, records in results.items():
        for record in records:
            by_id.setdefault(record["id"], {})[mode] = record

    lines = [
        "| ID | Category | Agents used | Exact (critic) | Judge (critic) | "
        "Exact (no critic) | Judge (no critic) | Revisions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for qid in sorted(by_id):
        with_c = by_id[qid].get("with_critic", {})
        without_c = by_id[qid].get("without_critic", {})
        agents = ", ".join(with_c.get("visited") or []) or "-"
        lines.append(
            f"| {qid} | {with_c.get('category', '-')} | {agents} | "
            f"{'yes' if with_c.get('exact_match') else 'NO'} | "
            f"{_fmt(with_c.get('judge_score'))} | "
            f"{'yes' if without_c.get('exact_match') else 'NO'} | "
            f"{_fmt(without_c.get('judge_score'))} | "
            f"{with_c.get('revisions', 0)} |"
        )
    return "\n".join(lines)


def failure_list(results: Dict[str, List[dict]]) -> str:
    """Every question the system got wrong — feeds the error analysis."""
    lines: List[str] = []
    for mode in MODES:
        failures = [
            r for r in results.get(mode, [])
            if not r.get("exact_match") or (r.get("judge_score") or 0) <= 3
        ]
        lines.append(f"\n### {MODE_LABELS.get(mode, mode)} — {len(failures)} failure(s)")
        if not failures:
            lines.append("\nNone.")
            continue
        for record in failures:
            lines.append(
                f"\n**{record['id']}** ({record.get('category')}) — "
                f"judge {record.get('judge_score')}, "
                f"exact match {'yes' if record.get('exact_match') else 'no'}"
            )
            lines.append(f"- Question: {record['question']}")
            lines.append(f"- Expected: {record['reference']}")
            lines.append(f"- Got: {record.get('answer') or '(nothing)'}")
            lines.append(f"- Agents used: {', '.join(record.get('visited') or []) or '-'}")
            lines.append(f"- Trace: {' -> '.join(record.get('steps') or [])}")
            if record.get("critic_reason"):
                lines.append(f"- Critic said: {record['critic_reason']}")
    return "\n".join(lines)


def write_report(
    results: Dict[str, List[dict]],
    summaries: Dict[str, ModeSummary],
    path: Optional[Path] = None,
) -> Path:
    """Write the full markdown report. Visual 4 of the final submission."""
    target = path or REPORT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y-%m-%d %H:%M")
    body = [
        "# Evaluation results — Multi-Agent AI Analyst (F11)",
        "",
        f"Generated {stamp}. Question set: `documents/eval_questions.json`.",
        "Every run used `use_memory=False` so the comparison is reproducible.",
        "",
        "## Metrics: with critic vs without critic",
        "",
        comparison_table(summaries),
        "",
        "## Per-question results",
        "",
        per_question_table(results),
        "",
        "## Failures",
        failure_list(results),
        "",
    ]
    target.write_text("\n".join(body), encoding="utf-8")
    return target


def evaluate_all(
    questions: Optional[List[EvalQuestion]] = None,
    modes: Tuple[str, ...] = MODES,
    force: bool = False,
    skip_ragas: bool = False,
    verbose: bool = True,
    use_proxy: bool = False,
) -> Tuple[Dict[str, List[dict]], Dict[str, ModeSummary]]:
    """Full harness: run, judge, RAGAS, summarise."""
    items = questions or load_questions()

    if verbose:
        print(f"\nPHASE 1 — running {len(items)} questions in {len(modes)} mode(s)")
    results = run_phase(items, modes=modes, force=force, verbose=verbose)

    if verbose:
        print("\nPHASE 2a — LLM judge")
    judge_phase(results, force=force, verbose=verbose, use_proxy=use_proxy)

    ragas_scores: Dict[str, dict] = {}
    if skip_ragas:
        if verbose:
            print("\nPHASE 2b — RAGAS skipped (--skip-ragas)")
    else:
        if verbose:
            print("\nPHASE 2b — RAGAS metrics")
        ragas_scores = ragas_phase(results, force=force, verbose=verbose, use_proxy=use_proxy)

    summaries = {
        mode: summarise(records, ragas_scores.get(mode))
        for mode, records in results.items()
        if records
    }
    return results, summaries