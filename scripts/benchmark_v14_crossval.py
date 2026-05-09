#!/usr/bin/env python3
"""
Benchmark V14: 5-Fold Cross-Validation with Balanced Sampling.

Runs 5-fold CV on benchmark_v7_samples.json (100 samples, 50/50 REAL/FAKE).
Each fold: 20 samples (10 REAL / 10 FAKE), no overlap.
Reports: Accuracy, F1, Precision, Recall, Standard Deviation per fold.

Usage:
    python scripts/benchmark_v14_crossval.py --output benchmarks/final/benchmark_v14_crossval_report.json
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# ── UTF-8 on Windows ──────────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ── Setup path ───────────────────────────────────────────────────────────────
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_SCRIPTS)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
os.environ.setdefault("LLM_PROVIDER", "google")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for _lib in ("httpx", "httpcore", "openai", "langchain", "urllib3", "trafilatura"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

logger = logging.getLogger("BENCHMARK_V14")

TKN_CHARS = 4  # rough characters-to-tokens ratio


# ─────────────────────────────────────────────────────────────────────────────
# Local Cache for Skip Duplicate Scrapes
# ─────────────────────────────────────────────────────────────────────────────


class ScrapeCache:
    """URL + content hash cache to skip duplicate scrapes."""

    def __init__(self) -> None:
        self._url_cache: dict[str, str] = {}
        self._content_cache: dict[str, list[str]] = {}
        self._query_cache: dict[str, list[str]] = {}

    def set_url(self, url: str, content: str) -> None:
        self._url_cache[url] = content[:5000]

    def get_url(self, url: str) -> str | None:
        return self._url_cache.get(url)

    def set_content_hash(self, content: str, result: list[str]) -> None:
        h = hashlib.md5(content.encode()).hexdigest()[:16]
        self._content_cache[h] = result

    def get_content_hash(self, content: str) -> list[str] | None:
        h = hashlib.md5(content.encode()).hexdigest()[:16]
        return self._content_cache.get(h)

    def set_query_results(self, query: str, results: list[str]) -> None:
        self._query_cache[query[:100]] = results

    def get_query_results(self, query: str) -> list[str] | None:
        return self._query_cache.get(query[:100])

    def stats(self) -> dict[str, int]:
        return {
            "url_cache_size": len(self._url_cache),
            "content_cache_size": len(self._content_cache),
            "query_cache_size": len(self._query_cache),
        }


_scrape_cache = ScrapeCache()


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SampleResult:
    sample_id: int = 0
    expected: str = "REAL"
    statement: str = ""

    multi_verdict: str | None = None
    multi_confidence: float = 0.0
    multi_correct: bool = False
    multi_latency: float = 0.0
    multi_tokens_in: int = 0
    multi_tokens_out: int = 0
    multi_error: str | None = None

    cache_hits: int = 0
    cache_misses: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Core async pipeline (same as benchmark_v8.py)
# ─────────────────────────────────────────────────────────────────────────────


async def run_multi_agent_async(
    text: str,
    batch_idx: int,
    ground_truth_evidence: str | None = None,
) -> dict[str, Any]:
    """Run the real TRUST orchestrator pipeline in a worker thread."""
    from trust_agents.orchestrator import run_trust_pipeline_sync

    start_time = time.perf_counter()
    result = await asyncio.to_thread(
        run_trust_pipeline_sync,
        text,
        10,
        False,
        ground_truth_evidence,
        True,
    )
    latency = time.perf_counter() - start_time

    summary = result.get("summary", {})
    verdict = summary.get("verdict", "UNCERTAIN")
    confidence = float(summary.get("average_confidence", 0.0) or 0.0)
    claims = result.get("claims", [])
    claim_results = result.get("results", [])

    tokens_in = len(text) // TKN_CHARS
    tokens_out = sum(len(str(r.get("reasoning", ""))) for r in claim_results) // TKN_CHARS

    return {
        "verdict": verdict,
        "confidence": confidence,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "claims_extracted": len(claims),
        "claims_processed": len(claim_results),
        "latency": latency,
    }


def _ground_truth_evidence_for_sample(sample: dict[str, Any]) -> str | None:
    """Use sample original_context as retrieval-side benchmark evidence."""
    context = sample.get("original_context")
    if isinstance(context, str) and context.strip():
        return context.strip()
    return None


async def _process_sample_async(sample: dict[str, Any], idx: int) -> SampleResult:
    """Process a single sample through the full V13/V14 pipeline."""
    sid = sample.get("sample_id", idx)
    expected = _normalize_label(sample.get("expected_label", "FAKE"))
    statement = sample.get("statement", "")

    r = SampleResult(
        sample_id=sid,
        expected=expected,
        statement=statement,
    )

    try:
        gt_evidence = _ground_truth_evidence_for_sample(sample)
        t0 = time.perf_counter()
        multi_resp = await run_multi_agent_async(
            statement,
            batch_idx=idx,
            ground_truth_evidence=gt_evidence,
        )

        r.multi_latency = multi_resp.get("latency", time.perf_counter() - t0)
        r.multi_verdict = multi_resp["verdict"]
        r.multi_confidence = multi_resp["confidence"]
        r.multi_tokens_in = multi_resp["tokens_in"]
        r.multi_tokens_out = multi_resp["tokens_out"]
        r.multi_correct = r.multi_verdict == expected

        logger.info(f"  [Fold sample] ID={sid} expected={expected} → {r.multi_verdict} ({r.multi_confidence:.1%}, {r.multi_latency:.1f}s)")
    except Exception as e:
        r.multi_error = str(e)
        logger.error(f"  ERROR sample {sid}: {e}")

    return r


async def process_batch_async(samples: list[dict[str, Any]]) -> list[SampleResult]:
    """Process all samples in a fold in parallel."""
    tasks = [_process_sample_async(sample, i) for i, sample in enumerate(samples)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    sample_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            sample = samples[i]
            sample_results.append(
                SampleResult(
                    sample_id=sample.get("sample_id", i),
                    expected=_normalize_label(sample.get("expected_label", "FAKE")),
                    statement=sample.get("statement", ""),
                    multi_error=str(result),
                )
            )
        else:
            sample_results.append(result)
    return sample_results


def _print_early_stop_debug(
    fold_idx: int,
    processed_results: list[SampleResult],
    projected_real_recall: float,
    projected_fake_recall: float,
    processed_real: int,
    processed_fake: int,
    total_real: int,
    total_fake: int,
) -> None:
    """Print debug table before aborting a fold early."""
    print("\n" + "=" * 90)
    print(f"EARLY STOP DEBUG — FOLD {fold_idx + 1}")
    print("=" * 90)
    print(f"  Processed REAL: {processed_real}/{total_real} | Processed FAKE: {processed_fake}/{total_fake} | Projected REAL Recall: {projected_real_recall:.1%} | Projected FAKE Recall: {projected_fake_recall:.1%}")
    print()
    print(f"  {'ID':<8} {'Expected':<10} {'Predicted':<10} {'Correct':<8} {'Conf':>7}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 7}")
    for result in processed_results:
        predicted = result.multi_verdict or "ERROR"
        correct = "YES" if result.multi_correct else "NO"
        print(f"  {result.sample_id:<8} {result.expected:<10} {predicted:<10} {correct:<8} {result.multi_confidence:>6.1%}")
    print("=" * 90)


def _compute_projected_recalls(
    processed_results: list[SampleResult],
    total_real: int,
    total_fake: int,
) -> dict[str, float]:
    """Compute projected recall ceilings for REAL and FAKE after partial fold progress."""
    processed_real = 0
    processed_fake = 0
    tp_real = 0
    tp_fake = 0

    for result in processed_results:
        if result.expected == "REAL":
            processed_real += 1
            if result.multi_verdict == "REAL":
                tp_real += 1
        elif result.expected == "FAKE":
            processed_fake += 1
            if result.multi_verdict == "FAKE":
                tp_fake += 1

    projected_real_recall = (tp_real + (total_real - processed_real)) / total_real if total_real else 1.0
    projected_fake_recall = (tp_fake + (total_fake - processed_fake)) / total_fake if total_fake else 1.0

    return {
        "processed_real": processed_real,
        "processed_fake": processed_fake,
        "tp_real": tp_real,
        "tp_fake": tp_fake,
        "projected_real_recall": projected_real_recall,
        "projected_fake_recall": projected_fake_recall,
    }


class EarlyStopFoldError(RuntimeError):
    """Raised when a fold cannot recover target recall under optimistic projection."""

    pass


def _normalize_label(label: str) -> str:
    """Normalize label to REAL/FAKE."""
    label_upper = label.upper().strip()
    if label_upper in ("REAL", "TRUE", "SUPPORTED"):
        return "REAL"
    if label_upper in ("FAKE", "FALSE", "REFUTED"):
        return "FAKE"
    return "UNCERTAIN"


# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold Cross-Validation Logic
# ─────────────────────────────────────────────────────────────────────────────


def create_balanced_folds(
    samples: list[dict[str, Any]],
    n_folds: int = 5,
    seed: int = 42,
) -> list[list[dict[str, Any]]]:
    """Split samples into N balanced folds (10 REAL / 10 FAKE each for n_folds=5).

    Works with any dataset that has even REAL/FAKE split.
    Returns n_folds lists of samples, each with 50/50 balance.
    """
    random.seed(seed)
    real_samples = [s for s in samples if _normalize_label(s.get("expected_label", "")) == "REAL"]
    fake_samples = [s for s in samples if _normalize_label(s.get("expected_label", "")) == "FAKE"]

    random.shuffle(real_samples)
    random.shuffle(fake_samples)

    real_per_fold = len(real_samples) // n_folds
    fake_per_fold = len(fake_samples) // n_folds

    folds = []
    for i in range(n_folds):
        real_start = i * real_per_fold
        fake_start = i * fake_per_fold
        fold = real_samples[real_start : real_start + real_per_fold] + fake_samples[fake_start : fake_start + fake_per_fold]
        random.shuffle(fold)  # shuffle within fold for run diversity
        folds.append(fold)

    return folds


def compute_fold_metrics(results: list[SampleResult]) -> dict[str, float]:
    """Compute accuracy, F1, precision, recall for a fold."""
    tp = fp = tn = fn = 0

    for r in results:
        pred = r.multi_verdict
        exp = r.expected

        if exp == "REAL":
            if pred == "REAL":
                tp += 1
            elif pred == "FAKE":
                fn += 1
            else:
                fn += 1  # UNCERTAIN treated as missed REAL
        elif exp == "FAKE":
            if pred == "FAKE":
                tn += 1
            elif pred == "REAL":
                fp += 1
            else:
                fp += 1  # UNCERTAIN treated as missed FAKE

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Also compute FAKE recall
    fake_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fake_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fake_f1 = 2 * fake_precision * fake_recall / (fake_precision + fake_recall) if (fake_precision + fake_recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy * 100, 1),
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1": round(f1 * 100, 1),
        "fake_precision": round(fake_precision * 100, 1),
        "fake_recall": round(fake_recall * 100, 1),
        "fake_f1": round(fake_f1 * 100, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


async def run_fold(
    fold_idx: int,
    fold_samples: list[dict[str, Any]],
    recall_target: float = 0.60,
    burn_in: int = 5,
    recall_target_real: float = 0.60,
    recall_target_fake: float = 0.50,
) -> dict[str, Any]:
    """Run a single fold with progressive recall kill-switch.

    After burn-in samples have been processed, projects optimistic ceiling
    for REAL and FAKE recall. Aborts if REAL recall cannot reach recall_target_real
    (default 95% for safety) even if ALL remaining samples were correct.
    FAKE recall target is tracked but not used for early stopping (default 70%).
    """
    total_real = sum(1 for s in fold_samples if _normalize_label(s.get("expected_label", "")) == "REAL")
    total_fake = sum(1 for s in fold_samples if _normalize_label(s.get("expected_label", "")) == "FAKE")

    logger.info("=" * 60)
    logger.info(f"FOLD {fold_idx + 1}/5: Running {len(fold_samples)} samples")
    logger.info(f"  REAL={total_real}, FAKE={total_fake}, recall_target={recall_target:.0%}")

    t0 = time.perf_counter()
    results = await process_batch_async(fold_samples)
    fold_time = time.perf_counter() - t0

    # ── Progressive Recall Forecast check ──────────────────────────────
    projected = _compute_projected_recalls(results, total_real, total_fake)
    projs = projected["projected_real_recall"]
    projs_fake = projected["projected_fake_recall"]

    # Log per-sample update after each result (for live tracking)
    # Check kill-switch after burn-in
    if len(results) >= burn_in:
        logger.info(f"  [Recall Forecast] burn-in={burn_in} | REAL: {projected['tp_real']}/{projected['processed_real']}/{total_real} (projected {projs:.1%}) | FAKE: {projected['tp_fake']}/{projected['processed_fake']}/{total_fake} (projected {projs_fake:.1%})")
        if projs < recall_target_real or (projs_fake < recall_target_fake and projs < recall_target_real):
            _print_early_stop_debug(
                fold_idx,
                results,
                projs,
                projs_fake,
                projected["processed_real"],
                projected["processed_fake"],
                total_real,
                total_fake,
            )
            logger.warning(f"  [KILL SWITCH] FOLD {fold_idx + 1}: REAL projected {projs:.1%} (target {recall_target_real:.0%}) | FAKE projected {projs_fake:.1%} (target {recall_target_fake:.0%}) → ABORT")
            raise EarlyStopFoldError(f"Fold {fold_idx + 1} projected REAL recall below {recall_target_real:.0%}. REAL {projs:.1%}, FAKE {projs_fake:.1%}")

    metrics = compute_fold_metrics(results)
    logger.info(f"FOLD {fold_idx + 1} done: Acc={metrics['accuracy']:.1f}% F1={metrics['f1']:.1f}% Recall={metrics['recall']:.1f}% FAKE-F1={metrics['fake_f1']:.1f}% ({metrics['tp']}TP/{metrics['tn']}TN/{metrics['fp']}FP/{metrics['fn']}FN, {fold_time:.1f}s)")

    return {
        "fold": fold_idx + 1,
        "metrics": metrics,
        "per_sample": [asdict(r) for r in results],
        "elapsed_seconds": round(fold_time, 1),
        "projected_recall_real": round(projs, 4),
        "projected_recall_fake": round(projs_fake, 4),
    }


async def run_crossval(
    samples: list[dict[str, Any]],
    n_folds: int = 5,
    recall_target: float = 0.60,
    burn_in: int = 5,
    recall_target_real: float = 0.60,
    recall_target_fake: float = 0.50,
) -> list[dict[str, Any]]:
    """Run all folds sequentially with rate-limit delay between folds.

    Each fold checked after burn_in samples: if projected REAL recall falls below
    recall_target_real, the fold is aborted. FAKE recall is tracked and reported
    but not used for early stopping.
    """
    folds = create_balanced_folds(samples, n_folds=n_folds)

    fold_results: list[dict[str, Any]] = []
    early_stopped = False
    for i, fold_samples in enumerate(folds):
        try:
            fold_result = await run_fold(
                i,
                fold_samples,
                recall_target=recall_target,
                burn_in=burn_in,
                recall_target_real=recall_target_real,
                recall_target_fake=recall_target_fake,
            )
            fold_results.append(fold_result)
        except EarlyStopFoldError as e:
            fold_results.append(
                {
                    "fold": i + 1,
                    "metrics": compute_fold_metrics([]),  # empty metrics
                    "per_sample": [],
                    "elapsed_seconds": 0.0,
                    "early_stop": True,
                    "early_stop_reason": str(e),
                }
            )
            early_stopped = True
            logger.error(f"  FOLD {i + 1} aborted: {e}")
            break  # stop entire cross-val

        # Rate-limit delay between folds
        if i < n_folds - 1:
            await asyncio.sleep(5.0)

    if early_stopped and len(fold_results) < n_folds:
        # Pad remaining folds with stub entries so the report is well-formed
        remaining = n_folds - len(fold_results)
        for _j in range(remaining):
            fold_results.append(
                {
                    "fold": len(fold_results) + 1,
                    "metrics": compute_fold_metrics([]),
                    "per_sample": [],
                    "elapsed_seconds": 0.0,
                    "early_stop": True,
                    "early_stop_reason": "fold not started (earlier fold aborted)",
                }
            )

    return fold_results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark V14: 5-Fold Cross-Validation with Balanced Sampling")
    parser.add_argument(
        "--json-file",
        default="benchmarks/history/benchmark_v7_samples.json",
        help="Path to benchmark samples JSON",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/final/benchmark_v14_crossval_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold splitting (default: 42)",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.60,
        help="Minimum projected recall threshold for kill-switch (default: 0.60)",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=5,
        help="Minimum samples before kill-switch activates (default: 5)",
    )

    args = parser.parse_args()

    # Load samples
    json_path = Path(args.json_file)
    if not json_path.exists():
        logger.error("Sample file not found: %s", json_path)
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        all_samples = json.load(f)

    real_count = sum(1 for s in all_samples if _normalize_label(s.get("expected_label", "")) == "REAL")
    fake_count = sum(1 for s in all_samples if _normalize_label(s.get("expected_label", "")) == "FAKE")
    logger.info(
        "Loaded %d samples (%d REAL, %d FAKE) from %s",
        len(all_samples),
        real_count,
        fake_count,
        json_path,
    )

    # Create folds
    folds = create_balanced_folds(all_samples, n_folds=args.folds, seed=args.seed)
    logger.info("Created %d balanced folds (%d samples each):", args.folds, len(folds[0]) if folds else 0)
    for i, fold in enumerate(folds):
        real_in = sum(1 for s in fold if _normalize_label(s.get("expected_label", "")) == "REAL")
        fake_in = sum(1 for s in fold if _normalize_label(s.get("expected_label", "")) == "FAKE")
        logger.info(f"  Fold {i + 1}: {len(fold)} samples ({real_in} REAL, {fake_in} FAKE)")

    # Run cross-validation
    logger.info("=" * 60)
    logger.info(f"Starting Benchmark V14: {args.folds}-Fold Cross-Validation")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 60)

    total_start = time.perf_counter()
    fold_results = asyncio.run(
        run_crossval(
            all_samples,
            n_folds=args.folds,
            recall_target=args.recall_target,
            burn_in=args.burn_in,
            recall_target_real=args.recall_target,
            recall_target_fake=args.recall_target,
        )
    )
    total_time = time.perf_counter() - total_start

    # Check for early stop outcome
    early_stops = [fr for fr in fold_results if fr.get("early_stop")]
    if early_stops:
        logger.warning(
            "Benchmark V14 EARLY STOP: %d fold(s) aborted before completion (%d/%d completed normally)",
            len(early_stops),
            sum(1 for fr in fold_results if not fr.get("early_stop")),
            len(fold_results),
        )

    # Aggregate metrics across folds
    all_metrics = [fr["metrics"] for fr in fold_results]

    def _stat(values: list[float]) -> dict[str, float]:
        if len(values) < 2:
            return {"mean": round(values[0], 2) if values else 0.0, "std": 0.0}
        return {"mean": round(mean(values), 2), "std": round(stdev(values), 2)}

    # Combined confusion matrix
    total_tp = sum(m["tp"] for m in all_metrics)
    total_tn = sum(m["tn"] for m in all_metrics)
    total_fp = sum(m["fp"] for m in all_metrics)
    total_fn = sum(m["fn"] for m in all_metrics)
    combined_total = total_tp + total_tn + total_fp + total_fn
    combined_accuracy = (total_tp + total_tn) / combined_total * 100 if combined_total else 0
    combined_precision = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) else 0
    combined_recall = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) else 0
    combined_f1 = 2 * combined_precision * combined_recall / (combined_precision + combined_recall) if (combined_precision + combined_recall) else 0

    combined_fake_precision = total_tn / (total_tn + total_fn) * 100 if (total_tn + total_fn) else 0
    combined_fake_recall = total_tn / (total_tn + total_fp) * 100 if (total_tn + total_fp) else 0
    combined_fake_f1 = 2 * combined_fake_precision * combined_fake_recall / (combined_fake_precision + combined_fake_recall) if (combined_fake_precision + combined_fake_recall) else 0

    report = {
        "version": "v14.3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_folds": args.folds,
        "samples_per_fold": len(folds[0]) if folds else 0,
        "total_samples": len(all_samples),
        "seed": args.seed,
        "recall_target": args.recall_target,
        "burn_in": args.burn_in,
        "early_stopped": len(early_stops) > 0,
        "total_time_seconds": round(total_time, 1),
        "combined": {
            "tp": total_tp,
            "tn": total_tn,
            "fp": total_fp,
            "fn": total_fn,
            "accuracy": round(combined_accuracy, 1),
            "precision": round(combined_precision, 1),
            "recall": round(combined_recall, 1),
            "f1": round(combined_f1, 1),
            "fake_precision": round(combined_fake_precision, 1),
            "fake_recall": round(combined_fake_recall, 1),
            "fake_f1": round(combined_fake_f1, 1),
        },
        "per_fold": [{"fold": fr["fold"], "metrics": fr["metrics"], "elapsed_seconds": fr["elapsed_seconds"]} for fr in fold_results],
        "fold_stats": {
            "accuracy": _stat([m["accuracy"] for m in all_metrics]),
            "precision": _stat([m["precision"] for m in all_metrics]),
            "recall": _stat([m["recall"] for m in all_metrics]),
            "f1": _stat([m["f1"] for m in all_metrics]),
            "fake_f1": _stat([m["fake_f1"] for m in all_metrics]),
            "fake_recall": _stat([m["fake_recall"] for m in all_metrics]),
        },
    }

    # Print summary
    print("\n" + "=" * 70)
    print("V14 CROSS-VALIDATION REPORT")
    print("=" * 70)
    print(f"Folds: {args.folds} | Samples/fold: {len(folds[0])} | Seed: {args.seed}")
    print(f"Total time: {total_time / 60:.1f} min ({total_time:.1f}s)")
    print()

    # Per-fold table
    print(f"  {'Fold':<6} {'Acc%':>6} {'Prec%':>6} {'Rec%':>6} {'F1%':>6} {'FAKE-F1%':>9} {'Time':>6}")
    print(f"  {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 9} {'-' * 6}")
    for fr in fold_results:
        m = fr["metrics"]
        print(f"  {fr['fold']:<6} {m['accuracy']:>5.1f}% {m['precision']:>5.1f}% {m['recall']:>5.1f}% {m['f1']:>5.1f}% {m['fake_f1']:>8.1f}% {fr['elapsed_seconds']:>5.0f}s")

    print()
    fs = report["fold_stats"]
    print(f"  {'Metric':<12} {'Mean%':>8} {'StdDev':>8}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 8}")
    print(f"  {'Accuracy':<12} {fs['accuracy']['mean']:>7.1f}% {fs['accuracy']['std']:>7.1f}")
    print(f"  {'Precision':<12} {fs['precision']['mean']:>7.1f}% {fs['precision']['std']:>7.1f}")
    print(f"  {'Recall':<12} {fs['recall']['mean']:>7.1f}% {fs['recall']['std']:>7.1f}")
    print(f"  {'F1':<12} {fs['f1']['mean']:>7.1f}% {fs['f1']['std']:>7.1f}")
    print(f"  {'FAKE-F1':<12} {fs['fake_f1']['mean']:>7.1f}% {fs['fake_f1']['std']:>7.1f}")
    print(f"  {'FAKE-Recall':<12} {fs['fake_recall']['mean']:>7.1f}% {fs['fake_recall']['std']:>7.1f}")

    print()
    print("Combined Confusion Matrix (all folds):")
    c = report["combined"]
    print(f"  TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")
    print(f"  Accuracy={c['accuracy']:.1f}% F1={c['f1']:.1f}% FAKE-F1={c['fake_f1']:.1f}% FAKE-Recall={c['fake_recall']:.1f}%")
    print()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Full report saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
