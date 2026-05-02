#!/usr/bin/env python3
"""
Debug script for TRUST Multi-Agent Pipeline.

Usage:
    uv run python scripts/debug_pipeline.py --text "Your text here"
    uv run python scripts/debug_pipeline.py --sample-id 0
    uv run python scripts/debug_pipeline.py --sample-id 0 --sample-id 1
    uv run python scripts/debug_pipeline.py --file benchmark_results.json --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv

# Ensure src/ is on the Python path for trust_agents imports
_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Setup verbose logging with UTF-8 support
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(name)-40s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DEBUG_PIPELINE")

# Reduce noise from some libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

load_dotenv()

# Set provider to google for debugging (avoid NVIDIA 502 issues)
os.environ.setdefault("LLM_PROVIDER", "google")


def load_sample_from_results(results_path: str, sample_id: int) -> dict[str, Any] | None:
    """Load a sample from benchmark results file."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    for result in data.get("results", []):
        for pred in result.get("predictions", []):
            if pred.get("id") == sample_id:
                return {
                    "claim": pred.get("claim", ""),
                    "true_label": pred.get("true_label", ""),
                    "sample_id": sample_id,
                }
    return None


def load_samples_from_vifactcheck(sample_ids: list[int]) -> list[dict[str, Any]]:
    """Load samples from ViFactCheck dataset by sample IDs."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Run: uv pip install datasets")
        return []

    dataset = load_dataset("tranthaihoa/vifactcheck", split="test")
    samples = []

    for i, item in enumerate(dataset):
        if i in sample_ids:
            label_raw = item.get("labels", item.get("label"))
            if isinstance(label_raw, int):
                label = "FAKE" if label_raw == 1 else "REAL"
            elif isinstance(label_raw, str):
                label = label_raw.strip().upper()
                if label == "TRUE":
                    label = "REAL"
                elif label == "FALSE":
                    label = "FAKE"
            else:
                label = "UNKNOWN"

            samples.append({
                "claim": item.get("Statement") or item.get("claim") or "",
                "evidence": item.get("Evidence") or item.get("evidence") or "",
                "true_label": label,
                "sample_id": i,
            })

    return samples


async def debug_stage_1_claim_extraction(text: str) -> list[str]:
    """
    Stage 1: Claim Extraction.
    Returns list of extracted claims.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: CLAIM EXTRACTION")
    logger.info("=" * 60)
    logger.info("Input text: %s", text[:200] + "..." if len(text) > 200 else text)

    try:
        from trust_agents.agents.claim_extractor import run_claim_extractor_agent
        from trust_agents.parsing import parse_claims_payload, extract_last_message_text

        logger.debug("Initializing Claim Extractor Agent...")
        claims = await run_claim_extractor_agent(text)

        logger.info("Extracted %d claims:", len(claims))
        for i, claim in enumerate(claims):
            logger.info("  [%d] %s", i, claim[:100] + "..." if len(claim) > 100 else claim)

        return claims

    except Exception as e:
        logger.error("Stage 1 FAILED: %s", e, exc_info=True)
        return []


async def debug_stage_2_evidence_retrieval(
    claim: str,
    ground_truth_evidence: str | None = None,
    use_gt_fallback: bool = True,
) -> list[dict[str, Any]]:
    """
    Stage 2: Evidence Retrieval.
    Returns list of evidence items.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: EVIDENCE RETRIEVAL")
    logger.info("=" * 60)
    logger.info("Query claim: %s", claim[:200] + "..." if len(claim) > 200 else claim)

    try:
        from trust_agents.agents.evidence_retrieval import run_evidence_retrieval_agent

        logger.debug("Initializing Evidence Retrieval Agent...")
        evidence = await run_evidence_retrieval_agent(
            claim,
            top_k=5,
            ground_truth_evidence=ground_truth_evidence,
            use_gt_fallback=use_gt_fallback,
        )

        logger.info("Retrieved %d evidence items:", len(evidence))
        for i, ev in enumerate(evidence[:3]):
            source = ev.get("source", "Unknown")
            score = ev.get("score", ev.get("hybrid_score", 0))
            text_preview = ev.get("text", "")[:100]
            logger.info("  [%d] Score: %.3f | Source: %s | Text: %s...",
                       i, score, source, text_preview)

        return evidence

    except Exception as e:
        logger.error("Stage 2 FAILED: %s", e, exc_info=True)
        return []


async def debug_stage_3_verification(
    claim: str,
    evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Stage 3: Verification.
    Returns verdict data.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: VERIFICATION")
    logger.info("=" * 60)
    logger.info("Claim: %s", claim[:200] + "..." if len(claim) > 200 else claim)
    logger.info("Evidence count: %d", len(evidence))

    try:
        from trust_agents.agents.verifier import run_verifier_agent

        logger.debug("Initializing Verifier Agent...")
        verdict_data = await run_verifier_agent(claim, evidence)

        logger.info("VERDICT: %s (confidence: %.1f%%)",
                   verdict_data.get("verdict", "unknown"),
                   verdict_data.get("confidence", 0) * 100)
        logger.info("Reasoning: %s", verdict_data.get("reasoning", "N/A")[:300])

        return verdict_data

    except Exception as e:
        logger.error("Stage 3 FAILED: %s", e, exc_info=True)
        return {
            "verdict": "uncertain",
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
            "error": str(e),
        }


async def debug_stage_4_explanation(
    claim: str,
    verdict_data: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Stage 4: Explanation.
    Returns final report.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: EXPLANATION")
    logger.info("=" * 60)

    try:
        from trust_agents.agents.explainer import run_explainer_agent

        logger.debug("Initializing Explainer Agent...")
        report = await run_explainer_agent(claim, verdict_data, evidence)

        logger.info("FINAL VERDICT: %s", report.get("verdict", "unknown"))
        logger.info("Summary: %s", report.get("summary", "N/A")[:200])

        return report

    except Exception as e:
        logger.error("Stage 4 FAILED: %s", e, exc_info=True)
        return {
            **verdict_data,
            "summary": f"Error generating explanation: {str(e)}",
            "error": str(e),
        }


async def run_full_pipeline_debug(
    text: str,
    expected_label: str = "",
    ground_truth_evidence: str | None = None,
) -> dict[str, Any]:
    """Run the full pipeline with detailed debugging."""
    logger.info("=" * 70)
    logger.info("STARTING FULL PIPELINE DEBUG")
    logger.info("Expected Label: %s", expected_label or "N/A")
    logger.info("=" * 70)

    start_time = time.time()

    # Stage 1: Claim Extraction
    claims = await debug_stage_1_claim_extraction(text)
    if not claims:
        logger.error("Pipeline FAILED at Stage 1: No claims extracted")
        return {
            "status": "failed",
            "stage": 1,
            "error": "No claims extracted",
            "elapsed_time": time.time() - start_time,
        }

    # Process ALL claims through remaining stages
    all_claims_results = []

    for idx, claim in enumerate(claims):
        logger.info("-" * 40)
        logger.info("PROCESSING CLAIM %d/%d", idx + 1, len(claims))
        logger.info("-" * 40)

        # Stage 2: Evidence Retrieval for this claim
        evidence = await debug_stage_2_evidence_retrieval(
            claim, ground_truth_evidence=ground_truth_evidence
        )

# Stage 3: Verification for this claim
        verdict_data = await debug_stage_3_verification(claim, evidence)

        # Normalize verdict to ensure it's either "true", "false", or "uncertain"
        raw_v = verdict_data.get("verdict", "uncertain").lower()
        if "false" in raw_v or "contradicted" in raw_v:
            v = "false"
        elif "true" in raw_v or "supported" in raw_v:
            v = "true"
        else:
            v = "uncertain"

        all_claims_results.append({
            "claim": claim,
            "evidence_count": len(evidence),
            "verdict": v,
            "confidence": verdict_data.get("confidence", 0),
            "reasoning": verdict_data.get("reasoning", ""),
        })

        logger.info("Claim %d verdict: %s (confidence: %.1f%%)",
                    idx + 1, v, verdict_data.get("confidence", 0) * 100)

    # Determine overall verdict logic:
    # - If ANY claim is FALSE → overall FAKE (clear contradiction found)
    # - If ANY claim is TRUE and none FALSE → overall REAL (claim confirmed)
    # - If ALL claims are UNCERTAIN → overall UNCERTAIN (insufficient evidence)
    has_false = any(c.get("verdict") == "false" for c in all_claims_results)
    has_true = any(c.get("verdict") == "true" for c in all_claims_results)
    all_uncertain = all(c.get("verdict") == "uncertain" for c in all_claims_results)

    if has_false:
        overall_verdict = "false"
    elif all_uncertain:
        overall_verdict = "uncertain"
    else:
        overall_verdict = "true"

    # Use first claim for explanation with aggregated verdict
    claim = claims[0]
    verdict_data = {
        "verdict": overall_verdict,
        "confidence": max(r["confidence"] for r in all_claims_results) if all_claims_results else 0.5,
        "reasoning": f"Processed {len(claims)} claims. Overall: {overall_verdict}. Details: " +
                     "; ".join([f"Claim {i+1}: {r['verdict']}" for i, r in enumerate(all_claims_results)])
    }

    # Stage 4: Explanation with aggregated verdict
    evidence = await debug_stage_2_evidence_retrieval(claim, ground_truth_evidence=ground_truth_evidence)
    report = await debug_stage_4_explanation(claim, verdict_data, evidence)

    elapsed = time.time() - start_time

    # Final summary
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED")
    logger.info("Expected Label: %s", expected_label)
    logger.info("Final Verdict (Aggregated): %s", overall_verdict)
    logger.info("Total Time: %.2fs", elapsed)
    logger.info("=" * 70)

    # Final verdict label normalization for results
    final_label = "FAKE" if overall_verdict == "false" else "REAL" if overall_verdict == "true" else "UNCERTAIN"

    return {
        "status": "success",
        "elapsed_time": elapsed,
        "claims_extracted": len(claims),
        "claims": claims,
        "evidence_count": len(evidence),
        "verdict": overall_verdict,
        "confidence": verdict_data.get("confidence", 0),
        "expected_label": expected_label,
        "verdict_label": final_label,
        "correct": expected_label.upper() == final_label if expected_label else None,
        "full_report": report,
        "all_claims_results": all_claims_results,
    }


async def run_single_agent_baseline(text: str, evidence_text: str = "") -> dict[str, Any]:
    """
    Run Single-Agent Baseline: One LLM call with Statement + Evidence.
    """
    logger.info("=" * 60)
    logger.info("SINGLE-AGENT BASELINE")
    logger.info("=" * 60)

    from trust_agents.llm.factory import create_chat_model
    model = create_chat_model()

    system_prompt = """Bạn là chuyên gia kiểm chứng thông tin (Fact-checker).
Nhiệm vụ của bạn là đánh giá một TUYÊN BỐ dựa trên các BẰNG CHỨNG cung cấp.

QUY TẮC:
1. FALSE: Nếu có mâu thuẫn trực tiếp.
2. TRUE: Nếu bằng chứng hỗ trợ hoặc không có mâu thuẫn rõ ràng.
3. UNCERTAIN: Nếu không liên quan hoặc không đủ thông tin.

Trả về JSON: {"verdict": "true|false|uncertain", "confidence": 0.0-1.0, "reasoning": "..."}"""

    prompt = f"""KIỂM CHỨNG TUYÊN BỐ SAU:

CLAIM: {text}

BẰNG CHỨNG (EVIDENCE):
{evidence_text if evidence_text else "Không có bằng chứng được cung cấp."}

Trả về CHỈ JSON:"""

    try:
        response = await model.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])

        content = response.content if hasattr(response, "content") else str(response)

        # Simple JSON parsing
        import json
        import re
        content = content.strip().replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(content)
        except:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            parsed = json.loads(match.group()) if match else {}

        return {
            "verdict": parsed.get("verdict", "uncertain"),
            "confidence": parsed.get("confidence", 0.5),
            "summary": parsed.get("reasoning", "No reasoning provided."),
            "detailed_explanation": parsed.get("reasoning", ""),
        }
    except Exception as e:
        logger.error("Single-Agent Baseline FAILED: %s", e)
        return {"verdict": "uncertain", "confidence": 0.0, "summary": str(e)}


async def main_async(args: argparse.Namespace) -> None:
    """Main async entry point."""
    samples: list[dict[str, Any]] = []

    if args.json_file:
        # Load samples from JSON file (hard_test_samples.json)
        with open(args.json_file, encoding="utf-8") as f:
            samples_raw = json.load(f)

        # Enrich samples with evidence from ViFactCheck if missing
        logger.info("Loading evidence for %d samples from ViFactCheck...", len(samples_raw))
        vifact_test = load_samples_from_vifactcheck([s["sample_id"] for s in samples_raw])
        vifact_map = {s["sample_id"]: s for s in vifact_test}

        for s in samples_raw:
            if s["sample_id"] in vifact_map:
                vifact_sample = vifact_map[s["sample_id"]]
                s["evidence"] = s.get("evidence") or vifact_sample.get("evidence")
                s["claim"] = s.get("statement") or s.get("claim") or vifact_sample.get("claim")

        samples = samples_raw
        logger.info("Loaded and enriched %d samples from %s", len(samples), args.json_file)

    elif args.text:
        # Single text input
        samples = [{"claim": args.text, "true_label": args.expected or ""}]
    elif args.sample_ids:
        # Load from ViFactCheck by sample IDs
        samples = load_samples_from_vifactcheck(args.sample_ids)
        if not samples:
            logger.error("No samples found for IDs: %s", args.sample_ids)
            return
    elif args.file and args.sample_id is not None:
        # Load from benchmark results file
        sample = load_sample_from_results(args.file, args.sample_id)
        if sample:
            samples = [sample]
        else:
            logger.error("Sample %d not found in %s", args.sample_id, args.file)
            return
    else:
        logger.error("Must provide either --json-file, --text, --sample-id, or --file with --sample-id")
        return

    results = []
    mode = args.mode or "multi"

    for i, sample in enumerate(samples):
        logger.info("\n")
        logger.info("*" * 70)
        logger.info("PROCESSING SAMPLE %d/%d (Mode: %s)", i + 1, len(samples), mode.upper())
        logger.info("*" * 70)

        start_time = time.time()

        if mode == "single":
            # Single-Agent Baseline
            statement = sample.get("statement") or sample.get("claim", "")
            evidence_text = sample.get("evidence", "")[:2000]  # Limit evidence length
            report = await run_single_agent_baseline(
                text=statement,
                evidence_text=evidence_text
            )
            elapsed = time.time() - start_time

            verdict_raw = report.get("verdict", "uncertain")
            verdict_label = "FAKE" if verdict_raw.lower() == "false" else ("REAL" if verdict_raw.lower() == "true" else "UNCERTAIN")
            expected = sample.get("expected_label", sample.get("true_label", ""))
            correct = expected.upper() == verdict_label if expected else None

            result = {
                "status": "success",
                "elapsed_time": elapsed,
                "verdict": verdict_raw,
                "verdict_label": verdict_label,
                "confidence": report.get("confidence", 0),
                "expected_label": expected,
                "correct": correct,
                "summary": report.get("summary", ""),
                "full_report": report,
                "mode": "single",
            }
            results.append(result)

            logger.info("SINGLE-AGENT VERDICT: %s (confidence: %.1f%%)", verdict_label, report.get("confidence", 0) * 100)
            logger.info("Expected: %s | Correct: %s", expected, "✓" if correct else "✗")

        else:
            # Multi-Agent (TRUST)
            statement = sample.get("statement") or sample.get("claim", "")
            result = await run_full_pipeline_debug(
                text=statement,
                expected_label=sample.get("expected_label", sample.get("true_label", "")),
                ground_truth_evidence=sample.get("evidence"),
            )
            result["mode"] = "multi"
            results.append(result)

            verdict_label = result.get("verdict_label", result.get("verdict", "unknown").upper())
            logger.info("MULTI-AGENT VERDICT: %s", verdict_label)

        # Rate limit protection
        if i < len(samples) - 1:
            logger.info("Waiting 30s before next sample (rate limit protection)...")
            time.sleep(30)

    # Summary
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("BENCHMARK SUMMARY (%s)", mode.upper())
    logger.info("=" * 70)

    correct_count = sum(1 for r in results if r.get("correct") == True)
    total = len(results)
    accuracy = correct_count / total * 100 if total > 0 else 0

    logger.info("Total samples: %d", total)
    logger.info("Correct: %d", correct_count)
    logger.info("Accuracy: %.1f%%", accuracy)
    logger.info("Average time per sample: %.1fs", sum(r.get("elapsed_time", 0) for r in results) / total if total > 0 else 0)

    for i, result in enumerate(results):
        sample = samples[i]
        verdict_label = result.get("verdict_label", result.get("verdict", "unknown"))
        expected = sample.get("expected_label", sample.get("true_label", "N/A"))
        correct = "✓" if result.get("correct") else "✗" if result.get("correct") is False else "?"

        logger.info("Sample %d: %s Verdict=%-10s Expected=%-10s Time=%.1fs %s",
                   i, result.get("mode", "?"), verdict_label, expected,
                   result.get("elapsed_time", 0), correct)

    # Save results
    if args.output:
        output_data = {
            "mode": mode,
            "benchmark_results": results,
            "samples": samples,
            "summary": {
                "total": len(results),
                "correct": correct_count,
                "accuracy": accuracy,
                "avg_time": sum(r.get("elapsed_time", 0) for r in results) / len(results) if results else 0,
            }
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to: %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TRUST Multi-Agent Pipeline vs Single-Agent Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-Agent (TRUST) on hard samples
  uv run python scripts/debug_pipeline.py --json-file hard_test_samples.json --mode multi --output benchmark_multi.json

  # Single-Agent (Baseline) on hard samples
  uv run python scripts/debug_pipeline.py --json-file hard_test_samples.json --mode single --output benchmark_single.json

  # Debug a single text
  uv run python scripts/debug_pipeline.py --text "Your Vietnamese text here"

  # Debug sample IDs from ViFactCheck
  uv run python scripts/debug_pipeline.py --sample-id 0 --sample-id 1
        """,
    )
    parser.add_argument("--text", help="Text to process")
    parser.add_argument("--expected", help="Expected label (REAL/FAKE) for validation")
    parser.add_argument("--sample-id", type=int, action="append", dest="sample_ids",
                        help="Sample ID from ViFactCheck dataset (can be repeated)")
    parser.add_argument("--file", help="Load sample from benchmark results JSON file")
    parser.add_argument("--file-sample-id", type=int, dest="sample_id",
                        help="Sample ID to load from file")
    parser.add_argument("--json-file", help="Load samples from JSON file (for batch benchmark)")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi",
                        help="Mode: single (baseline) or multi (TRUST multi-agent)")
    parser.add_argument("--output", "-o", help="Save benchmark results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()