#!/usr/bin/env python3
"""
Benchmark V8: Optimized Latency - Target < 30 mins for 100 samples.

Optimizations:
1. Parallel Processing: Batch size = 5 samples concurrently
2. Async/Await: Full async pipeline for LLM calls
3. Local Cache: Skip duplicate scrapes using URL/content hash
4. Reduced Rate Limits: Only delay on cache misses

Usage:
    python scripts/benchmark_v8.py --batch-size 5 --max-samples 100
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
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

from dotenv import load_dotenv  # noqa: E402 - must be after sys.path setup

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

logger = logging.getLogger("BENCHMARK_V8")

# ─────────────────────────────────────────────────────────────────────────────
# Local Cache for Skip Duplicate Scrapes
# ─────────────────────────────────────────────────────────────────────────────

class ScrapeCache:
    """Cache to skip duplicate URL scrapes."""

    def __init__(self):
        self._url_cache: dict[str, str] = {}  # url_hash -> content_hash
        self._content_cache: dict[str, str] = {}  # content_hash -> content
        self._query_cache: dict[str, list[str]] = {}  # query -> list of content

    def get_url_content(self, url: str) -> str | None:
        """Get cached content for URL."""
        url_hash = self._hash_url(url)
        if url_hash in self._url_cache:
            content_hash = self._url_cache[url_hash]
            return self._content_cache.get(content_hash)
        return None

    def set_url_content(self, url: str, content: str) -> None:
        """Cache URL content."""
        url_hash = self._hash_url(url)
        content_hash = self._hash_content(content)
        self._url_cache[url_hash] = content_hash
        self._content_cache[content_hash] = content

    def get_query_results(self, query: str) -> list[str] | None:
        """Get cached search results for query."""
        return self._query_cache.get(query.lower().strip())

    def set_query_results(self, query: str, results: list[str]) -> None:
        """Cache search results for query."""
        self._query_cache[query.lower().strip()] = results

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.md5(content.encode()[:5000].encode()).hexdigest()[:16]

    def stats(self) -> dict[str, int]:
        return {
            "url_cache_size": len(self._url_cache),
            "content_cache_size": len(self._content_cache),
            "query_cache_size": len(self._query_cache),
        }


# Global cache instances
_scrape_cache = ScrapeCache()

# Import semantic cache from orchestrator
_orchestrator_cache_available = False
try:
    from trust_agents.orchestrator import (
        _cache_result,
        _claim_cache,
        _get_cached_result,
        _get_claim_signature,
        clear_claim_cache,
    )
    _orchestrator_cache_available = True
except ImportError:
    pass


def _check_semantic_cache(claim: str) -> tuple[dict | None, int, int]:
    """Check semantic cache and return (result, hits, misses)."""
    if not _orchestrator_cache_available:
        return None, 0, 1

    cached = _get_cached_result(claim)
    if cached is not None:
        return cached.copy(), 1, 0
    return None, 0, 1


def _store_semantic_cache(claim: str, result: dict) -> None:
    """Store result in semantic cache."""
    if _orchestrator_cache_available:
        _cache_result(claim, result)


# ─────────────────────────────────────────────────────────────────────────────
# Metric dataclasses
# ─────────────────────────────────────────────────────────────────────────────

TKN_CHARS = 4  # rough estimate: 1 Vietnamese token ≈ 4 chars


@dataclass
class SampleResult:
    sample_id: int
    expected: str
    statement: str

    # Multi Agent (primary focus)
    multi_verdict: str | None = None
    multi_correct: bool | None = None
    multi_latency: float = 0.0
    multi_tokens_in: int = 0
    multi_tokens_out: int = 0
    multi_error: str | None = None

    # Cache stats
    cache_hits: int = 0
    cache_misses: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Async LLM Factory
# ─────────────────────────────────────────────────────────────────────────────

_async_model = None


async def get_async_model():
    """Get async LLM model instance."""
    global _async_model
    if _async_model is None:
        from trust_agents.llm.factory import create_chat_model
        _async_model = create_chat_model()
    return _async_model


# ─────────────────────────────────────────────────────────────────────────────
# Async Single Agent
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_PROMPT = """Bạn là chuyên gia kiểm chứng thông tin (Fact-checker).
Nhiệm vụ: Đánh giá một TUYÊN BỐ dựa trên các BẰNG CHỨNG cung cấp.

QUY TẮC BẮT BUỘC:
1. FALSE: Evidence MÂU THUẨN TRỰC TIẾP với claim hoặc IM LẶNG về con số cụ thể.
2. TRUE: Evidence CHỦ ĐỘNG XÁC NHẬN claim.
3. UNCERTAIN: Không liên quan hoặc thiếu thông tin.

Trả về CHỈ JSON: {"verdict": "true|false|uncertain", "confidence": 0.0-1.0, "reasoning": "..."}"""


async def run_single_agent_async(statement: str, snippets: list[str]) -> dict[str, Any]:
    """One-shot LLM check using search snippets as evidence."""
    evidence_text = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets[:5]))
    prompt = f"""KIỂM CHỨNG TUYÊN BỐ:

CLAIM: {statement}

EVIDENCE:
{evidence_text}

Trả về CHỈ JSON:"""

    model = await get_async_model()
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

    label_map = {"true": "REAL", "false": "FAKE", "uncertain": "UNCERTAIN"}
    label = label_map.get(verdict.lower(), "UNCERTAIN")

    tokens_in = len(prompt) // TKN_CHARS
    tokens_out = len(content) // TKN_CHARS

    return {"verdict": label, "confidence": confidence, "tokens_in": tokens_in, "tokens_out": tokens_out}


# ─────────────────────────────────────────────────────────────────────────────
# Async Web Search with Cache
# ─────────────────────────────────────────────────────────────────────────────

async def search_snippets_async(query: str, num: int = 3) -> tuple[list[str], int, int]:
    """Get search snippets with caching."""
    global _scrape_cache

    # Check cache first
    cached = _scrape_cache.get_query_results(query)
    if cached is not None:
        return cached[:num], 1, 0  # hits, misses

    try:
        from trust_agents.rag.web_search import search_web

        results = search_web(query, num_results=num, use_cache=True)
        snippets = [r.get("content", r.get("snippet", "")) for r in results if r]

        # Cache results
        if snippets:
            _scrape_cache.set_query_results(query, snippets)

        return snippets, 0, 1  # hits, misses
    except Exception as e:
        logger.debug("Search failed for %s: %s", query[:50], e)
        return [], 0, 1


# ─────────────────────────────────────────────────────────────────────────────
# Async Multi-Agent with Parallel Claim Processing
# ─────────────────────────────────────────────────────────────────────────────

async def run_multi_agent_async(
    text: str,
    batch_idx: int,
) -> dict[str, Any]:
    """Async TRUST multi-agent pipeline with parallel claim processing."""
    from trust_agents.agents.claim_extractor import run_claim_extractor_agent
    from trust_agents.rag.web_search import clear_search_cache

    start_time = time.perf_counter()

    # Step 1: Extract claims (async)
    clear_search_cache()
    claims = await run_claim_extractor_agent(text)
    if not claims:
        claims = [text]

    logger.info(f"  [Batch {batch_idx}] Extracted {len(claims)} claims")

    # Process claims in parallel batches
    claim_results = []
    claim_tasks = []

    for claim_idx, claim in enumerate(claims):
        # Create task for each claim
        task = _process_single_claim_async(
            claim=claim,
            claim_idx=claim_idx,
            batch_idx=batch_idx,
        )
        claim_tasks.append(task)

    # Execute all claim tasks in parallel
    claim_results = await asyncio.gather(*claim_tasks, return_exceptions=True)

    # Filter out exceptions
    valid_results = []
    for result in claim_results:
        if isinstance(result, Exception):
            logger.warning(f"  [Batch {batch_idx}] Claim processing error: {result}")
            valid_results.append({
                "verdict": "uncertain",
                "confidence": 0.3,
                "reasoning": f"Error: {str(result)}",
            })
        else:
            valid_results.append(result)

    # Calculate aggregate verdict (simplified V4)
    verdict = _aggregate_verdicts_v4(valid_results)

    latency = time.perf_counter() - start_time
    tokens_in = len(text) // TKN_CHARS
    tokens_out = sum(len(str(r.get("reasoning", ""))) for r in valid_results) // TKN_CHARS

    return {
        "verdict": verdict,
        "confidence": sum(r.get("confidence", 0) for r in valid_results) / max(len(valid_results), 1),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "claims_extracted": len(claims),
        "claims_processed": len(valid_results),
        "latency": latency,
    }


async def _process_single_claim_async(
    claim: str,
    claim_idx: int,
    batch_idx: int,
) -> dict[str, Any]:
    """Process a single claim through retrieval + verification."""
    from trust_agents.agents.verifier import run_verifier_agent
    from trust_agents.rag.content_extractor import extract_content_batch_async
    from trust_agents.rag.web_search import clear_search_cache, search_web

    global _scrape_cache

    # 1. Check semantic cache (fast - avoids network I/O)
    sem_result, sem_hits, sem_misses = _check_semantic_cache(claim)
    if sem_result is not None:
        return sem_result

    # 2. Check scrape cache
    cached_content = _scrape_cache.get_query_results(claim)
    cache_hit = cached_content is not None

    if not cache_hit:
        # Search for evidence
        clear_search_cache()
        search_results = search_web(claim, num_results=7, use_cache=True)

        if search_results:
            urls = [r.get("url", "") for r in search_results if r.get("url")]
            if urls:
                content_results = await extract_content_batch_async(urls, query=claim, max_chars=8000)
                if content_results:
                    # Cache the content
                    all_content = "\n\n".join([c.get("content", "") for c in content_results])
                    _scrape_cache.set_query_results(claim, [all_content])
                    cached_content = [all_content]

    if cached_content:
        # Verify claim with cached content
        evidence = [{"content": c, "source": "cached"} for c in cached_content]
        result = await run_verifier_agent(claim, evidence)
        # 3. Store in semantic cache for future reuse
        _store_semantic_cache(claim, result)
        return result

    # Fallback: uncertain
    return {
        "verdict": "uncertain",
        "confidence": 0.3,
        "reasoning": "No evidence found for claim",
    }


def _aggregate_verdicts_v4(results: list[dict[str, Any]]) -> str:
    """Aggregate claim verdicts into final verdict (V4 weighted scoring)."""
    if not results:
        return "UNCERTAIN"

    # Simple aggregation (can be enhanced with source authority)
    real_count = sum(1 for r in results if r.get("verdict") in ("true", "supported"))
    fake_count = sum(1 for r in results if r.get("verdict") in ("false", "contradicted"))

    if fake_count >= 2:
        return "FAKE"
    if fake_count == 1:
        fake_high_conf = any(
            r.get("confidence", 0) >= 0.85
            for r in results
            if r.get("verdict") in ("false", "contradicted")
        )
        if fake_high_conf:
            return "FAKE"
    if real_count > fake_count:
        return "REAL"
    return "UNCERTAIN"


# ─────────────────────────────────────────────────────────────────────────────
# Parallel Batch Processing
# ─────────────────────────────────────────────────────────────────────────────

async def process_batch_async(
    samples: list[dict[str, Any]],
    batch_size: int,
    start_idx: int,
) -> list[SampleResult]:
    """Process a batch of samples in parallel."""
    tasks = []

    for i, sample in enumerate(samples):
        task = _process_sample_async(sample, start_idx + i)
        tasks.append(task)

    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert to SampleResult objects
    sample_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            sample = samples[i]
            logger.error(f"Sample {i} failed: {result}")
            sample_results.append(SampleResult(
                sample_id=sample.get("sample_id", start_idx + i),
                expected=_normalize_label(sample.get("expected_label", "FAKE")),
                statement=sample.get("statement", ""),
                multi_error=str(result),
            ))
        else:
            sample_results.append(result)

    return sample_results


async def _process_sample_async(sample: dict[str, Any], idx: int) -> SampleResult:
    """Process a single sample through the full pipeline."""
    sid = sample.get("sample_id", idx)
    expected = _normalize_label(sample.get("expected_label", "FAKE"))
    statement = sample.get("statement", "")

    if not statement:
        return SampleResult(
            sample_id=sid,
            expected=expected,
            statement=statement,
            multi_error="Empty statement",
        )

    logger.info(f"Processing sample {idx + 1}: ID={sid}, expected={expected}")

    r = SampleResult(
        sample_id=sid,
        expected=expected,
        statement=statement,
    )

    try:
        # Run multi-agent pipeline (async)
        t0 = time.perf_counter()
        multi_resp = await run_multi_agent_async(statement, batch_idx=idx)

        r.multi_latency = multi_resp.get("latency", time.perf_counter() - t0)
        r.multi_verdict = multi_resp["verdict"]
        r.multi_tokens_in = multi_resp["tokens_in"]
        r.multi_tokens_out = multi_resp["tokens_out"]
        r.multi_correct = r.multi_verdict == expected

        logger.info(
            f"  → {r.multi_verdict} ({multi_resp['confidence']:.1%}, {r.multi_latency:.1f}s, "
            f"claims={multi_resp['claims_processed']})"
        )
    except Exception as e:
        r.multi_error = str(e)
        logger.error(f"  ERROR: {e}")

    return r


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_label(label: str) -> str:
    """Normalize label to REAL/FAKE."""
    label_upper = label.upper().strip()
    if label_upper in ("REAL", "TRUE", "SUPPORTED"):
        return "REAL"
    if label_upper in ("FAKE", "FALSE", "REFUTED"):
        return "FAKE"
    return "UNCERTAIN"


# ─────────────────────────────────────────────────────────────────────────────
# Main Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_benchmark_async(
    samples: list[dict[str, Any]],
    batch_size: int = 5,
    inter_batch_delay: float = 2.0,
) -> list[SampleResult]:
    """
    Run A/B benchmark on all samples with parallel batch processing.

    Args:
        samples: List of benchmark samples
        batch_size: Number of samples to process in parallel
        inter_batch_delay: Delay between batches (rate-limit protection)
    """
    results: list[SampleResult] = []
    total_samples = len(samples)

    logger.info(f"Starting benchmark with batch_size={batch_size}, total={total_samples}")

    # Process in batches
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_samples = samples[batch_start:batch_end]

        logger.info("=" * 60)
        logger.info(f"Processing batch {batch_start // batch_size + 1}: samples {batch_start + 1}-{batch_end}")

        batch_start_time = time.perf_counter()

        # Process batch in parallel
        batch_results = await process_batch_async(batch_samples, batch_size, batch_start)
        results.extend(batch_results)

        batch_elapsed = time.perf_counter() - batch_start_time

        # Log batch stats
        cache_stats = _scrape_cache.stats()
        logger.info(
            f"Batch completed in {batch_elapsed:.1f}s "
            f"(avg {batch_elapsed / len(batch_samples):.1f}s per sample)"
        )
        logger.info(f"Cache stats: {cache_stats}")

        # Rate-limit delay between batches
        if batch_end < total_samples:
            logger.info(f"Waiting {inter_batch_delay}s before next batch...")
            await asyncio.sleep(inter_batch_delay)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(results: list[SampleResult], total_time: float) -> dict[str, Any]:
    """Compute summary statistics and generate the final report."""
    n = len(results)
    multi_correct = sum(1 for r in results if r.multi_correct is True)
    multi_errors = sum(1 for r in results if r.multi_error)

    multi_latencies = [r.multi_latency for r in results if r.multi_latency > 0]
    multi_tokens = [
        r.multi_tokens_in + r.multi_tokens_out
        for r in results if r.multi_tokens_in > 0
    ]

    real_count = sum(1 for r in results if r.expected == "REAL")
    fake_count = sum(1 for r in results if r.expected == "FAKE")

    report = {
        "version": "v10",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": n,
        "real_count": real_count,
        "fake_count": fake_count,
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_sample": round(total_time / n, 1) if n else 0,
        "summary": {
            "multi_agent": {
                "accuracy": round(multi_correct / n * 100, 1) if n else 0,
                "correct": multi_correct,
                "errors": multi_errors,
                "avg_latency_s": round(sum(multi_latencies) / len(multi_latencies), 2) if multi_latencies else 0,
                "total_tokens_approx": sum(multi_tokens),
                "avg_tokens_approx": round(sum(multi_tokens) / len(multi_tokens)) if multi_tokens else 0,
            },
        },
        "cache_stats": _scrape_cache.stats(),
        "per_sample_results": [asdict(r) for r in results],
    }

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark V8: Optimized Latency (< 30 mins for 100 samples)"
    )
    parser.add_argument(
        "--json-file",
        default="benchmark_v7_samples.json",
        help="Path to benchmark samples JSON",
    )
    parser.add_argument(
        "--output",
        default="benchmark_v10_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of samples to process in parallel (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max samples to run (default: 100)",
    )
    parser.add_argument(
        "--inter-delay",
        type=float,
        default=2.0,
        help="Seconds between batches for rate-limit protection (default: 2)",
    )

    args = parser.parse_args()

    # Load samples
    json_path = Path(args.json_file)
    if not json_path.exists():
        logger.error("Sample file not found: %s", json_path)
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        samples = json.load(f)

    samples = samples[: args.limit]
    logger.info("Loaded %d samples from %s", len(samples), json_path)

    # Run benchmark
    logger.info("=" * 60)
    logger.info("Starting Benchmark V8 (Optimized Latency)")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Inter-batch delay: {args.inter_delay}s")
    logger.info("=" * 60)

    start_time = time.perf_counter()

    # Run async benchmark
    results = asyncio.run(run_benchmark_async(
        samples,
        batch_size=args.batch_size,
        inter_batch_delay=args.inter_delay,
    ))

    total_time = time.perf_counter() - start_time

    logger.info("=" * 60)
    logger.info("Benchmark V8 completed in %.1f seconds (%.1f minutes)", total_time, total_time / 60)

    # Generate and save report
    report = generate_report(results, total_time)

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    m = report["summary"]["multi_agent"]

    print("\n" + "=" * 60)
    print("BENCHMARK V8 REPORT (Optimized Latency)")
    print("=" * 60)
    print(f"Total Samples:    {report['total_samples']} ({report['real_count']} REAL, {report['fake_count']} FAKE)")
    print(f"Total Time:       {report['total_time_seconds']:.1f}s ({report['total_time_seconds']/60:.1f} min)")
    print(f"Avg per Sample:   {report['avg_time_per_sample']:.1f}s")
    print()
    header = f"  {'Mode':<12} {'Accuracy':>10} {'Correct':>8} {'Errors':>7} {'Avg Latency':>12}"
    print(header)
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*7} {'-'*12}")
    result_line = (
        f"  {'Multi-Agent':<12} {m['accuracy']:>9.1f}% "
        f"{m['correct']:>8} {m['errors']:>7} {m['avg_latency_s']:>11.1f}s"
    )
    print(result_line)
    print()
    print(f"  Cache Stats: {report['cache_stats']}")
    print()
    print(f"  Full report saved to: {out_path}")
    print("=" * 60)

    # Check if target met
    target_minutes = 30
    actual_minutes = total_time / 60
    if actual_minutes <= target_minutes:
        print(f"✓ Target achieved: {actual_minutes:.1f} min <= {target_minutes} min")
    else:
        print(f"✗ Target missed: {actual_minutes:.1f} min > {target_minutes} min")


if __name__ == "__main__":
    main()
