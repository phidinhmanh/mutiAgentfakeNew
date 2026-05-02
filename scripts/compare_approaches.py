#!/usr/bin/env python3
"""
Compare Multi-Agent Pipeline vs Single LLM approach for fake news detection.

Usage:
    uv run python scripts/compare_approaches.py --sample-id 0 --sample-id 1
    uv run python scripts/compare_approaches.py --text "Your text here"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("COMPARE")

load_dotenv()


async def single_llm_fact_check(text: str) -> dict[str, Any]:
    """
    Single LLM approach: One prompt to verify the entire text.
    Simpler but less granular analysis.
    """
    start_time = time.time()

    try:
        from trust_agents.llm.factory import create_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        model = create_chat_model()

        system_prompt = """Bạn là một chuyên gia kiểm tra thông tin (fact-checker) chuyên nghiệp.
Nhiệm vụ của bạn là xác minh tính xác thực của các tuyên bố trong văn bản tiếng Việt.

Hãy phân tích văn bản và đưa ra kết luận:
- REAL: Thông tin đáng tin cậy, có thể xác minh được
- FAKE: Thông tin giả mạo hoặc sai sự thật
- UNCERTAIN: Không đủ thông tin để kết luận

Trả lời CHỈ bằng JSON format:
{
    "verdict": "REAL|FAKE|UNCERTAIN",
    "confidence": 0.0-1.0,
    "reasoning": "giải thích ngắn gọn"
}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Văn bản cần kiểm tra:\n{text}"),
        ]

        response = await model.ainvoke(messages)
        result_text = response.content if hasattr(response, "content") else str(response)

        # Parse JSON response
        import re
        json_match = re.search(r"\{[^}]+\}", result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"verdict": "UNCERTAIN", "confidence": 0.5, "reasoning": "Parse failed"}

        return {
            "approach": "single_llm",
            "verdict": result.get("verdict", "UNCERTAIN"),
            "confidence": result.get("confidence", 0.5),
            "reasoning": result.get("reasoning", ""),
            "elapsed_time": time.time() - start_time,
            "status": "success",
        }

    except Exception as e:
        return {
            "approach": "single_llm",
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
            "elapsed_time": time.time() - start_time,
            "status": "failed",
            "error": str(e),
        }


async def multi_agent_fact_check(text: str) -> dict[str, Any]:
    """
    Multi-Agent approach: Pipeline với 4 agents riêng biệt.
    - Claim Extraction: Tách các tuyên bố
    - Evidence Retrieval: Tìm bằng chứng
    - Verifier: Xác minh từng tuyên bố
    - Explainer: Tạo báo cáo
    """
    start_time = time.time()

    try:
        from trust_agents.orchestrator import run_trust_pipeline_sync

        result = run_trust_pipeline_sync(text, skip_evidence=False)

        return {
            "approach": "multi_agent",
            "verdict": result.get("summary", {}).get("verdict", "UNCERTAIN"),
            "confidence": result.get("summary", {}).get("confidence", 0.5),
            "claims_count": len(result.get("claims", [])),
            "results": result.get("results", []),
            "elapsed_time": time.time() - start_time,
            "status": "success",
        }

    except Exception as e:
        return {
            "approach": "multi_agent",
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
            "elapsed_time": time.time() - start_time,
            "status": "failed",
            "error": str(e),
        }


async def compare_on_sample(
    text: str,
    expected_label: str,
    sample_id: int,
) -> dict[str, Any]:
    """Compare both approaches on a single sample."""
    logger.info("=" * 60)
    logger.info(f"SAMPLE {sample_id}: {text[:80]}...")
    logger.info(f"Expected: {expected_label}")
    logger.info("=" * 60)

    # Run both approaches concurrently
    single_task = single_llm_fact_check(text)
    multi_task = multi_agent_fact_check(text)

    single_result, multi_result = await asyncio.gather(single_task, multi_task)

    # Determine correctness
    single_correct = (
        expected_label.upper() == single_result.get("verdict", "").upper()
        if expected_label and single_result.get("status") == "success"
        else None
    )
    multi_correct = (
        expected_label.upper() == multi_result.get("verdict", "").upper()
        if expected_label and multi_result.get("status") == "success"
        else None
    )

    return {
        "sample_id": sample_id,
        "text": text,
        "expected_label": expected_label,
        "single_llm": single_result,
        "multi_agent": multi_result,
        "single_correct": single_correct,
        "multi_correct": multi_correct,
    }


async def main_async(args: argparse.Namespace) -> None:
    samples: list[dict[str, Any]] = []

    if args.text:
        samples = [{"text": args.text, "expected": args.expected or "", "id": 0}]
    elif args.sample_ids:
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets not installed. Run: uv pip install datasets")
            return

        dataset = load_dataset("tranthaihoa/vifactcheck", split="test")
        for idx in args.sample_ids:
            item = dataset[idx]
            label_raw = item.get("labels", item.get("label"))
            if isinstance(label_raw, int):
                label = "FAKE" if label_raw == 1 else "REAL"
            else:
                label = str(label_raw).upper()
                if label == "TRUE":
                    label = "REAL"

            samples.append({
                "text": item.get("statement", item.get("claim", "")),
                "expected": label,
                "id": idx,
            })

    if not samples:
        logger.error("No samples provided")
        return

    results = []
    for i, sample in enumerate(samples):
        result = await compare_on_sample(
            text=sample["text"],
            expected_label=sample["expected"],
            sample_id=sample["id"],
        )
        results.append(result)

        if i < len(samples) - 1:
            await asyncio.sleep(30)  # Rate limit

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)

    single_success = sum(1 for r in results if r["single_llm"].get("status") == "success")
    multi_success = sum(1 for r in results if r["multi_agent"].get("status") == "success")
    single_correct = sum(1 for r in results if r.get("single_correct") == True)
    multi_correct = sum(1 for r in results if r.get("multi_correct") == True)

    logger.info("Single LLM:  %d/%d samples processed, %d correct",
               single_success, len(results), single_correct)
    logger.info("Multi-Agent: %d/%d samples processed, %d correct",
               multi_success, len(results), multi_correct)

    avg_single_time = sum(
        r["single_llm"].get("elapsed_time", 0) for r in results
    ) / max(len(results), 1)
    avg_multi_time = sum(
        r["multi_agent"].get("elapsed_time", 0) for r in results
    ) / max(len(results), 1)

    logger.info("Average time - Single LLM: %.1fs, Multi-Agent: %.1fs",
                avg_single_time, avg_multi_time)

    # Per-sample results
    for r in results:
        s_verdict = r["single_llm"].get("verdict", "N/A")
        m_verdict = r["multi_agent"].get("verdict", "N/A")
        s_correct = "✓" if r.get("single_correct") else "✗" if r.get("single_correct") is False else "?"
        m_correct = "✓" if r.get("multi_correct") else "✗" if r.get("multi_correct") is False else "?"

        logger.info("Sample %d: Expected=%s | Single=%s %s | Multi=%s %s",
                   r["sample_id"], r["expected_label"],
                   s_verdict, s_correct, m_verdict, m_correct)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"results": results, "summary": {
                "total": len(results),
                "single_success": single_success,
                "multi_success": multi_success,
                "single_correct": single_correct,
                "multi_correct": multi_correct,
            }}, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Single LLM vs Multi-Agent approaches")
    parser.add_argument("--text", help="Text to check")
    parser.add_argument("--expected", help="Expected label (REAL/FAKE)")
    parser.add_argument("--sample-id", type=int, action="append", dest="sample_ids", help="Sample IDs")
    parser.add_argument("--output", "-o", help="Output JSON file")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
