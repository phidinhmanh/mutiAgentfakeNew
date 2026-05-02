#!/usr/bin/env python3
"""Compare Multi-Agent vs Single-Agent benchmark results."""

import json
import sys
from collections import defaultdict

def load_benchmark(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_verdict(v: str) -> str:
    """Map internal verdicts to FAKE/REAL/UNCERTAIN."""
    v = (v or "").lower()
    if v == "false":
        return "FAKE"
    if v == "true":
        return "REAL"
    return "UNCERTAIN"


def analyze(results: list, samples: list) -> dict:
    """Compute metrics for a benchmark run."""
    correct = 0
    by_expected = defaultdict(lambda: {"correct": 0, "total": 0})
    by_predicted = defaultdict(lambda: {"correct": 0, "total": 0})

    for result, sample in zip(results, samples):
        expected = sample.get("expected_label", "").upper()
        predicted = normalize_verdict(result.get("verdict", ""))

        is_correct = expected == predicted
        if is_correct:
            correct += 1

        by_expected[expected]["total"] += 1
        if is_correct:
            by_expected[expected]["correct"] += 1

        by_predicted[predicted]["total"] += 1
        if is_correct:
            by_predicted[predicted]["correct"] += 1

    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0
    avg_time = sum(r.get("elapsed_time", 0) for r in results) / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time": avg_time,
        "by_expected": dict(by_expected),
        "by_predicted": dict(by_predicted),
    }


def main():
    multi_path = "benchmark_multi.json"
    single_path = "benchmark_single.json"

    print("=" * 70)
    print("BENCHMARK COMPARISON: Multi-Agent (TRUST) vs Single-Agent")
    print("=" * 70)

    # Load data
    try:
        multi_data = load_benchmark(multi_path)
        single_data = load_benchmark(single_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run both benchmarks first:")
        print(f"  uv run python scripts/debug_pipeline.py --json-file hard_test_samples.json --mode multi --output {multi_path}")
        print(f"  uv run python scripts/debug_pipeline.py --json-file hard_test_samples.json --mode single --output {single_path}")
        sys.exit(1)

    multi_results = multi_data.get("benchmark_results", [])
    single_results = single_data.get("benchmark_results", [])
    samples = multi_data.get("samples", single_data.get("samples", []))

    print(f"\nLoaded {len(multi_results)} multi-agent results, {len(single_results)} single-agent results")

    # Analyze
    multi_stats = analyze(multi_results, samples)
    single_stats = analyze(single_results, samples)

    # Summary table
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Multi-Agent':>15} {'Single-Agent':>15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {multi_stats['accuracy']:>14.1f}% {single_stats['accuracy']:>14.1f}%")
    print(f"{'Correct/Total':<25} {multi_stats['correct']:>15}/{multi_stats['total']} {single_stats['correct']:>15}/{single_stats['total']}")
    print(f"{'Avg Time (s)':<25} {multi_stats['avg_time']:>15.1f} {single_stats['avg_time']:>15.1f}")
    print("-" * 70)

    diff = multi_stats["accuracy"] - single_stats["accuracy"]
    winner = "Multi-Agent" if diff > 0 else "Single-Agent" if diff < 0 else "Tie"
    print(f"Winner: {winner} (+{abs(diff):.1f}%)")

    # Per-category breakdown
    print("\n" + "=" * 70)
    print("ACCURACY BY EXPECTED LABEL")
    print("=" * 70)
    print(f"{'Label':<15} {'Multi-Agent':>15} {'Single-Agent':>15} {'Diff':>10}")
    print("-" * 70)
    all_labels = set(multi_stats["by_expected"].keys()) | set(single_stats["by_expected"].keys())
    for label in sorted(all_labels):
        m = multi_stats["by_expected"].get(label, {"correct": 0, "total": 0})
        s = single_stats["by_expected"].get(label, {"correct": 0, "total": 0})
        m_acc = m["correct"] / m["total"] * 100 if m["total"] > 0 else 0
        s_acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        d = m_acc - s_acc
        print(f"{label:<15} {m_acc:>14.1f}% {s_acc:>14.1f}% {d:>+9.1f}%")

    # Per-predicted breakdown
    print("\n" + "=" * 70)
    print("VERDICT DISTRIBUTION")
    print("=" * 70)
    print(f"{'Predicted':<15} {'Multi-Agent':>15} {'Single-Agent':>15}")
    print("-" * 70)
    all_preds = set(multi_stats["by_predicted"].keys()) | set(single_stats["by_predicted"].keys())
    for pred in sorted(all_preds):
        m = multi_stats["by_predicted"].get(pred, {"total": 0})["total"]
        s = single_stats["by_predicted"].get(pred, {"total": 0})["total"]
        print(f"{pred:<15} {m:>15} {s:>15}")

    # Sample-by-sample comparison
    print("\n" + "=" * 70)
    print("SAMPLE-BY-SAMPLE COMPARISON")
    print("=" * 70)
    print(f"{'#':<3} {'Expected':<10} {'Multi-Agent':<12} {'Single-Agent':<12} {'Result'}")
    print("-" * 70)

    for i, sample in enumerate(samples):
        expected = sample.get("expected_label", "").upper()
        m_pred = normalize_verdict(multi_results[i].get("verdict", ""))
        s_pred = normalize_verdict(single_results[i].get("verdict", ""))

        m_correct = "[OK]" if expected == m_pred else "[X]"
        s_correct = "[OK]" if expected == s_pred else "[X]"

        both_correct = (expected == m_pred) and (expected == s_pred)
        both_wrong = (expected != m_pred) and (expected != s_pred)

        result = "TIE"
        if expected == m_pred and expected != s_pred:
            result = "MULTI WIN"
        elif expected != m_pred and expected == s_pred:
            result = "SINGLE WIN"

        print(f"{i:<3} {expected:<10} {m_pred:<10}{m_correct} {s_pred:<10}{s_correct} {result}")

    # Key insights
    multi_win = sum(
        1 for i, sample in enumerate(samples)
        if sample.get("expected_label", "").upper() == normalize_verdict(multi_results[i].get("verdict", ""))
        and sample.get("expected_label", "").upper() != normalize_verdict(single_results[i].get("verdict", ""))
    )
    single_win = sum(
        1 for i, sample in enumerate(samples)
        if sample.get("expected_label", "").upper() != normalize_verdict(multi_results[i].get("verdict", ""))
        and sample.get("expected_label", "").upper() == normalize_verdict(single_results[i].get("verdict", ""))
    )

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print(f"- Multi-Agent wins on {multi_win} samples where Single-Agent fails")
    print(f"- Single-Agent wins on {single_win} samples where Multi-Agent fails")
    print(f"- Multi-Agent avg time: {multi_stats['avg_time']:.1f}s vs Single-Agent: {single_stats['avg_time']:.1f}s")


if __name__ == "__main__":
    main()