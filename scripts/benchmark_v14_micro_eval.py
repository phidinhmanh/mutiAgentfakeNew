#!/usr/bin/env python3
"""
Micro-Eval: Rerun benchmark on only 14 FAKE->REAL failures
to validate the V14 Contradiction Overrides GT fix.

Usage:
    python scripts/benchmark_v14_micro_eval.py
"""

import asyncio
import io
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_SCRIPTS)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
os.environ.setdefault("LLM_PROVIDER", "google")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for _lib in ("httpx", "httpcore", "openai", "langchain", "urllib3", "trafilatura"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

logger = logging.getLogger("MICRO_EVAL")

TKN_CHARS = 4


@dataclass
class SampleResult:
    sample_id: int = 0
    expected: str = "FAKE"
    statement: str = ""

    multi_verdict: str | None = None
    multi_confidence: float = 0.0
    multi_correct: bool = False
    multi_latency: float = 0.0
    multi_tokens_in: int = 0
    multi_tokens_out: int = 0
    multi_error: str | None = None


async def run_multi_agent_async(
    text: str,
    ground_truth_evidence: str | None = None,
) -> dict[str, Any]:
    """Run the real TRUST orchestrator pipeline in a worker thread."""
    from trust_agents.orchestrator import run_trust_pipeline_sync

    t0 = time.perf_counter()
    result = await asyncio.to_thread(
        run_trust_pipeline_sync, text, 10, False, ground_truth_evidence, True,
    )
    latency = time.perf_counter() - t0

    summary = result.get("summary", {})
    verdict = summary.get("verdict", "UNCERTAIN")
    confidence = float(summary.get("average_confidence", 0.0) or 0.0)
    claims = result.get("claims", [])
    claim_results = result.get("results", [])

    return {
        "verdict": verdict,
        "confidence": confidence,
        "tokens_in": len(text) // TKN_CHARS,
        "tokens_out": sum(len(str(r.get("reasoning", ""))) for r in claim_results) // TKN_CHARS,
        "claims_extracted": len(claims),
        "claims_processed": len(claim_results),
        "latency": latency,
    }


def _normalize_label(label: str) -> str:
    label_upper = label.upper().strip()
    if label_upper in ("REAL", "TRUE", "SUPPORTED"):
        return "REAL"
    if label_upper in ("FAKE", "FALSE", "REFUTED"):
        return "FAKE"
    return "UNCERTAIN"


async def process_one(sample: dict[str, Any]) -> SampleResult:
    sid = sample.get("sample_id", 0)
    expected = _normalize_label(sample.get("expected_label", "FAKE"))
    statement = sample.get("statement", "")

    r = SampleResult(sample_id=sid, expected=expected, statement=statement)

    try:
        gt = sample.get("original_context", "")
        if isinstance(gt, str) and gt.strip():
            # Pass GT as ground_truth_evidence
            multi_resp = await run_multi_agent_async(statement, ground_truth_evidence=gt.strip())
        else:
            multi_resp = await run_multi_agent_async(statement)

        r.multi_latency = multi_resp["latency"]
        r.multi_verdict = multi_resp["verdict"]
        r.multi_confidence = multi_resp["confidence"]
        r.multi_tokens_in = multi_resp["tokens_in"]
        r.multi_tokens_out = multi_resp["tokens_out"]
        r.multi_correct = r.multi_verdict == expected

        logger.info(
            f"[Micro] ID={sid} expected={expected} -> {r.multi_verdict} "
            f"({r.multi_confidence:.1%}, {r.multi_latency:.1f}s) "
            f"CORRECT={r.multi_correct}"
        )
    except Exception as e:
        r.multi_error = str(e)
        logger.error(f"[Micro] ID={sid} ERROR: {e}")

    return r


async def main() -> None:
    # Load dataset
    data = json.load(open("benchmarks/history/benchmark_v7_samples.json", encoding="utf-8"))

    # The 14 FAKE->REAL errors from V14 cross-val
    ids_to_fix = [
        164, 5558, 3502, 5334, 6161, 447, 3850, 1800,
        3542, 1618, 1463, 4097, 3652, 3799,
    ]

    samples = [s for s in data if s["sample_id"] in ids_to_fix]
    # Deduplicate by sample_id
    seen = set()
    unique = []
    for s in samples:
        if s["sample_id"] not in seen:
            seen.add(s["sample_id"])
            unique.append(s)
    samples = unique

    logger.info(f"Running micro-eval on {len(samples)} FAKE samples (should be 14)")

    results = []
    for sample in samples:
        # Small delay between samples for rate-limit
        r = await process_one(sample)
        results.append(r)
        if len(results) < len(samples):
            await asyncio.sleep(3.0)

    # Score
    correct = sum(1 for r in results if r.multi_correct)
    accuracy = correct / len(results) * 100 if results else 0

    fp = sum(1 for r in results if r.expected == "FAKE" and r.multi_verdict == "REAL")
    fn = sum(1 for r in results if r.expected == "REAL" and r.multi_verdict == "FAKE")
    tn = sum(1 for r in results if r.expected == "FAKE" and r.multi_verdict == "FAKE")
    tp = sum(1 for r in results if r.expected == "REAL" and r.multi_verdict == "REAL")
    unc = sum(1 for r in results if r.multi_verdict == "UNCERTAIN")

    print("\n" + "=" * 80)
    print("V14 MICRO-EVAL: 14 FAKE->REAL FAILURES")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(results)} returned to FAKE)")
    print(f"True Negative (FAKE correctly caught): {tn}")
    print(f"False Positive (still REAL): {fp}")
    print(f"UNCERTAIN (no longer REAL): {unc}")
    print()

    for r in results:
        mark = "SUCCESS" if r.multi_correct else "FAIL"
        if r.multi_verdict == "UNCERTAIN" and r.expected == "FAKE":
            mark = "BETTER"  # UNCERTAIN is better than REAL for FAKE
        print(
            f"  ID={r.sample_id:5d} | expected={r.expected:5s} -> {r.multi_verdict:12s} "
            f"({r.multi_confidence:.1%}) | {mark}"
        )

    print()
    print(f"Summary: {tn} correct FAKE, {fp} still FAKE->REAL, {unc} FAKE->UNCERTAIN (improved)")
    print(f"Goal: >= 7/14 returned to FAKE (50%+) → {'PASS' if correct >= 7 else 'UNDER TARGET'}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())