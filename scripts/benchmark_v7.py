#!/usr/bin/env python3
"""
Benchmark V7: Large-Scale A/B Validation - Single Agent vs. Multi-Agent Pipeline.

Compares:
  - Single Agent: Search snippets + one LLM call (baseline approach)
  - Multi Agent:  TRUTH pipeline with claim extraction, parallel scraping, verification

Usage:
    uv run python scripts/benchmark_v7.py
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── UTF-8 on Windows ──────────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ── Setup path ────────────────────────────────────────────────────────────────
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_SCRIPTS)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_PROVIDER", "google")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-38s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for _lib in ("httpx", "httpcore", "openai", "langchain", "urllib3", "trafilatura"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

logger = logging.getLogger("BENCHMARK_V7")

# ─────────────────────────────────────────────────────────────────────────────
# Metric dataclasses
# ─────────────────────────────────────────────────────────────────────────────

TKN_CHARS = 4  # rough estimate: 1 Vietnamese token ≈ 4 chars


@dataclass
class SampleResult:
    sample_id: int
    expected: str
    statement: str

    # Single Agent
    single_verdict: str | None = None
    single_correct: bool | None = None
    single_latency: float = 0.0
    single_tokens_in: int = 0
    single_tokens_out: int = 0
    single_error: str | None = None

    # Multi Agent
    multi_verdict: str | None = None
    multi_correct: bool | None = None
    multi_latency: float = 0.0
    multi_tokens_in: int = 0
    multi_tokens_out: int = 0
    multi_error: str | None = None

    # Summary
    agree: bool = False
    single_only_correct: bool = False
    multi_only_correct: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Single-Agent: snippet-based baseline
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_PROMPT = """Bạn là chuyên gia kiểm chứng thông tin (Fact-checker).
Nhiệm vụ: Đánh giá một TUYÊN BỐ dựa trên các BẰNG CHỨNG cung cấp.

QUY TẮC BẮT BUỘC:
1. FALSE: Evidence MÂU THUẪN TRỰC TIẾP với claim hoặc IM LẶNG về con số cụ thể.
2. TRUE: Evidence CHỦ ĐỘNG XÁC NHẬN claim.
3. UNCERTAIN: Không liên quan hoặc thiếu thông tin.

Trả về CHỈ JSON: {"verdict": "true|false|uncertain", "confidence": 0.0-1.0, "reasoning": "..."}"""


async def run_single_agent(statement: str, snippets: list[str]) -> dict[str, Any]:
    """One-shot LLM check using search snippets as evidence."""
    from trust_agents.llm.factory import create_chat_model

    evidence_text = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets[:5]))
    prompt = f"""KIỂM CHỨNG TUYÊN BỐ:

CLAIM: {statement}

EVIDENCE:
{evidence_text}

Trả về CHỈ JSON:"""

    model = create_chat_model()
    resp = await model.ainvoke(
        [{"role": "system", "content": SINGLE_PROMPT}, {"role": "user", "content": prompt}]
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}

    verdict = parsed.get("verdict", "uncertain")
    confidence = float(parsed.get("confidence", 0.5))

    # Map to FAKE/REAL/UNCERTAIN
    label_map = {"true": "REAL", "false": "FAKE", "uncertain": "UNCERTAIN"}
    label = label_map.get(verdict.lower(), "UNCERTAIN")

    tokens_in = len(prompt) // TKN_CHARS
    tokens_out = len(content) // TKN_CHARS

    return {"verdict": label, "confidence": confidence, "tokens_in": tokens_in, "tokens_out": tokens_out}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent: full TRUST pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_agent(text: str) -> dict[str, Any]:
    """Full TRUST multi-agent pipeline with parallel scraping."""
    from trust_agents.orchestrator import TRUSTOrchestrator

    orchestrator = TRUSTOrchestrator(top_k_evidence=5, max_claim_workers=3)

    # Capture token usage via size estimate of inputs/outputs
    # (LLM factory doesn't expose token counters, use char/4 estimate)
    result = orchestrator.process_text(text)
    summary = result.summary

    verdict = summary.get("verdict", "UNCERTAIN")
    if verdict.upper() in ("REAL", "FAKE", "UNCERTAIN"):
        label = verdict.upper()
    else:
        label = "UNCERTAIN"

    # Estimate tokens from result size
    tokens_in = len(text) // TKN_CHARS  # Claim + evidence inputs
    tokens_out = len(str(summary.get("reasoning", ""))) // TKN_CHARS

    return {
        "verdict": label,
        "confidence": summary.get("average_confidence", 0.5),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "claims_extracted": len(result.claims),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Web search (for Single Agent snippets)
# ─────────────────────────────────────────────────────────────────────────────

def search_snippets(query: str, num: int = 3) -> list[str]:
    """Get search snippets for Single Agent evidence."""
    try:
        from trust_agents.rag.web_search import search_web

        results = search_web(query, num_results=num, use_cache=True)
        return [r.get("content", r.get("snippet", "")) for r in results if r]
    except Exception as e:
        logger.debug("Search failed for %s: %s", query[:50], e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_label(label: str) -> str:
    """Normalize label to REAL/FAKE."""
    l = label.upper().strip()
    if l in ("REAL", "TRUE", "SUPPORTED"):
        return "REAL"
    if l in ("FAKE", "FALSE", "REFUTED"):
        return "FAKE"
    return "UNCERTAIN"


def run_benchmark(samples: list[dict[str, Any]], rate_limit_delay: float = 30.0) -> list[SampleResult]:
    """
    Run A/B benchmark on all samples.
    rate_limit_delay: seconds to wait between samples (DDG protection).
    """
    results: list[SampleResult] = []
    errors: dict[str, int] = {}

    for i, sample in enumerate(samples):
        idx = i + 1
        sid = sample.get("sample_id", idx)
        expected = _normalize_label(sample.get("expected_label", "FAKE"))
        statement = sample.get("statement", "")
        if not statement:
            logger.warning("Sample %d has empty statement, skipping", sid)
            continue

        logger.info("─" * 60)
        logger.info("Sample %d/%d (ID %s) — Expected: %s", idx, len(samples), sid, expected)
        logger.info("Statement: %s", statement[:120] + "...")

        r = SampleResult(
            sample_id=sid,
            expected=expected,
            statement=statement,
        )

        # ── Single Agent ──────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            snippets = search_snippets(statement, num=3)
            logger.info("  [SINGLE] Got %d snippets", len(snippets))

            single_resp = asyncio.run(run_single_agent(statement, snippets))
            r.single_latency = time.perf_counter() - t0
            r.single_verdict = single_resp["verdict"]
            r.single_tokens_in = single_resp["tokens_in"]
            r.single_tokens_out = single_resp["tokens_out"]
            r.single_correct = r.single_verdict == expected
            logger.info(
                "  [SINGLE] → %s (%.1f%%, %.1fs, ~%d tokens in, ~%d tokens out)",
                r.single_verdict,
                single_resp["confidence"] * 100,
                r.single_latency,
                r.single_tokens_in,
                r.single_tokens_out,
            )
        except Exception as e:
            r.single_error = str(e)
            logger.error("  [SINGLE] ERROR: %s", e)

        # ── Multi Agent ───────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            multi_resp = run_multi_agent(statement)
            r.multi_latency = time.perf_counter() - t0
            r.multi_verdict = multi_resp["verdict"]
            r.multi_tokens_in = multi_resp["tokens_in"]
            r.multi_tokens_out = multi_resp["tokens_out"]
            r.multi_correct = r.multi_verdict == expected
            logger.info(
                "  [MULTI]  → %s (%.1f%%, %.1fs, claims=%d, ~%d tokens in, ~%d tokens out)",
                r.multi_verdict,
                multi_resp["confidence"] * 100,
                r.multi_latency,
                multi_resp["claims_extracted"],
                r.multi_tokens_in,
                r.multi_tokens_out,
            )
        except Exception as e:
            r.multi_error = str(e)
            logger.error("  [MULTI]  ERROR: %s", e)

        # ── Agreement & crossover analysis ────────────────────────────────
        if r.single_verdict and r.multi_verdict:
            r.agree = r.single_verdict == r.multi_verdict
        r.single_only_correct = r.single_correct and not r.multi_correct
        r.multi_only_correct = r.multi_correct and not r.single_correct

        results.append(r)

        # Rate-limit protection (30s between samples for DDG)
        if idx < len(samples):
            logger.info("  Waiting %.0fs (rate-limit protection)...", rate_limit_delay)
            time.sleep(rate_limit_delay)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(results: list[SampleResult]) -> dict[str, Any]:
    """Compute summary statistics and generate the final report."""
    n = len(results)
    single_correct = sum(1 for r in results if r.single_correct is True)
    multi_correct = sum(1 for r in results if r.multi_correct is True)
    single_errors = sum(1 for r in results if r.single_error)
    multi_errors = sum(1 for r in results if r.multi_error)
    agreements = sum(1 for r in results if r.agree)
    single_only = sum(1 for r in results if r.single_only_correct)
    multi_only = sum(1 for r in results if r.multi_only_correct)

    single_latencies = [r.single_latency for r in results if r.single_latency > 0]
    multi_latencies = [r.multi_latency for r in results if r.multi_latency > 0]
    single_tokens = [
        r.single_tokens_in + r.single_tokens_out
        for r in results
        if r.single_tokens_in > 0
    ]
    multi_tokens = [
        r.multi_tokens_in + r.multi_tokens_out for r in results if r.multi_tokens_in > 0
    ]

    # Crossover: multi wrong but single right
    multi_missed = [r for r in results if r.single_only_correct]

    report = {
        "version": "v7",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": n,
        "real_count": sum(1 for r in results if r.expected == "REAL"),
        "fake_count": sum(1 for r in results if r.expected == "FAKE"),
        "summary": {
            "single_agent": {
                "accuracy": round(single_correct / n * 100, 1) if n else 0,
                "correct": single_correct,
                "errors": single_errors,
                "avg_latency_s": round(sum(single_latencies) / len(single_latencies), 2) if single_latencies else 0,
                "total_tokens_approx": sum(single_tokens),
                "avg_tokens_approx": round(sum(single_tokens) / len(single_tokens)) if single_tokens else 0,
            },
            "multi_agent": {
                "accuracy": round(multi_correct / n * 100, 1) if n else 0,
                "correct": multi_correct,
                "errors": multi_errors,
                "avg_latency_s": round(sum(multi_latencies) / len(multi_latencies), 2) if multi_latencies else 0,
                "total_tokens_approx": sum(multi_tokens),
                "avg_tokens_approx": round(sum(multi_tokens) / len(multi_tokens)) if multi_tokens else 0,
            },
        },
        "comparison": {
            "agreement_rate": round(agreements / n * 100, 1) if n else 0,
            "single_only_correct_count": single_only,
            "multi_only_correct_count": multi_only,
            "latency_ratio": round(
                (sum(single_latencies) / len(single_latencies))
                / (sum(multi_latencies) / len(multi_latencies))
                if single_latencies and multi_latencies
                else 0,
                2,
            ),
            "token_ratio": round(
                (sum(single_tokens) / len(single_tokens))
                / (sum(multi_tokens) / len(multi_tokens))
                if single_tokens and multi_tokens
                else 0,
                2,
            ),
        },
        "crossover_analysis": {
            "description": "Samples where Single Agent was correct but Multi-Agent was wrong",
            "count": len(multi_missed),
            "samples": [
                {
                    "sample_id": r.sample_id,
                    "expected": r.expected,
                    "single_verdict": r.single_verdict,
                    "multi_verdict": r.multi_verdict,
                    "statement": r.statement[:200],
                }
                for r in multi_missed
            ],
        },
        "per_sample_results": [asdict(r) for r in results],
    }

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark V7: Single Agent vs. Multi-Agent A/B Validation"
    )
    parser.add_argument(
        "--json-file",
        default="benchmark_v7_samples.json",
        help="Path to benchmark samples JSON (default: benchmark_v7_samples.json)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_v7_report.json",
        help="Output report path (default: benchmark_v7_report.json)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=30.0,
        help="Seconds between samples for DDG rate-limit protection (default: 30)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max samples to run (default: 100)",
    )
    args = parser.parse_args()

    # Load samples
    json_path = Path(args.json_file)
    if not json_path.exists():
        logger.error("Sample file not found: %s", json_path)
        logger.error("Run: python scripts/benchmark_v7.py --json-file hard_test_samples.json")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        samples = json.load(f)

    samples = samples[: args.limit]
    logger.info("Loaded %d samples from %s", len(samples), json_path)

    # Run benchmark
    logger.info("Starting Benchmark V7 A/B Validation")
    logger.info("=" * 60)

    start = time.perf_counter()
    results = run_benchmark(samples, rate_limit_delay=args.delay)
    elapsed = time.perf_counter() - start

    logger.info("=" * 60)
    logger.info("Benchmark V7 completed in %.1f seconds", elapsed)

    # Generate and save report
    report = generate_report(results)

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    s = report["summary"]["single_agent"]
    m = report["summary"]["multi_agent"]
    c = report["comparison"]

    print("\n" + "=" * 60)
    print("BENCHMARK V7 REPORT")
    print("=" * 60)
    print(f"Total Samples:    {report['total_samples']} ({report['real_count']} REAL, {report['fake_count']} FAKE)")
    print()
    print(f"  {'Mode':<12} {'Accuracy':>10} {'Correct':>8} {'Errors':>7} {'Avg Latency':>12} {'Avg Tokens':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*7} {'-'*12} {'-'*12}")
    print(f"  {'Single Agent':<12} {s['accuracy']:>9.1f}% {s['correct']:>8} {s['errors']:>7} {s['avg_latency_s']:>11.1f}s {s['avg_tokens_approx']:>11,}")
    print(f"  {'Multi-Agent':<12} {m['accuracy']:>9.1f}% {m['correct']:>8} {m['errors']:>7} {m['avg_latency_s']:>11.1f}s {m['avg_tokens_approx']:>11,}")
    print()
    print(f"  Agreement Rate: {c['agreement_rate']:.1f}%")
    print(f"  Latency Ratio (Single/Multi): {c['latency_ratio']:.2f}x")
    print(f"  Token Ratio (Single/Multi):   {c['token_ratio']:.2f}x")
    print()
    print(f"  Crossover (Single correct, Multi wrong): {c['single_only_correct_count']}")
    print(f"  Crossover (Multi correct, Single wrong): {c['multi_only_correct_count']}")
    print()
    print(f"  Full report saved to: {out_path}")
    print("=" * 60)

    # Print error samples
    if report["crossover_analysis"]["count"] > 0:
        print("\nCrossover Analysis (Multi missed, Single correct):")
        for s in report["crossover_analysis"]["samples"]:
            print(f"  ID {s['sample_id']}: Expected={s['expected']}, Single={s['single_verdict']}, Multi={s['multi_verdict']}")
            print(f"    Statement: {s['statement'][:150]}...")


if __name__ == "__main__":
    main()