"""
Explainer Agent - Direct LLM approach for generating explanations.

Creates natural language explanations of verification results with evidence citations.
"""

import json
import logging
from typing import Any

from dotenv import load_dotenv

from trust_agents.llm.factory import create_chat_model

load_dotenv()
logger = logging.getLogger("Explainer.Agent")


async def run_explainer_agent(
    claim: str,
    verdict_data: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate explanation for verification result."""
    model = create_chat_model()

    logger.info("[AGENT] Explainer Agent (Direct LLM) initialized")

    verdict = verdict_data.get("verdict", "uncertain")
    confidence = verdict_data.get("confidence", 0.5)
    reasoning = verdict_data.get("reasoning", "")

    # Format evidence for context
    evidence_text = "\n\n".join([
        f"Source {i+1} ({item.get('url', 'no link')}):\n{item.get('text', str(item))[:300]}..."
        for i, item in enumerate(evidence[:5])
    ])

    system_prompt = """Bạn là chuyên gia giải thích thông tin (Explainer).
Nhiệm vụ: Tạo báo cáo giải thích kết quả kiểm chứng một cách dễ hiểu, khách quan.
Báo cáo cần có:
1. Kết luận (Verdict)
2. Tóm tắt ngắn gọn
3. Giải thích chi tiết dựa trên bằng chứng
4. Trích dẫn nguồn

Trả về JSON: {
    "verdict": "true|false|uncertain",
    "confidence": 0.0-1.0,
    "summary": "...",
    "detailed_explanation": "...",
    "citations": [{"source_idx": 1, "text": "..."}]
}"""

    prompt = f"""Tạo báo cáo giải thích cho kết quả sau:

CLAIM: {claim}
VERDICT: {verdict}
CONFIDENCE: {confidence:.2f}
REASONING: {reasoning}

BẰNG CHỨNG:
{evidence_text}

Trả về CHỈ JSON:"""

    try:
        response = await model.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])

        content = response.content if hasattr(response, "content") else str(response)
        logger.info(f"[AGENT] LLM response length: {len(content)}")

        # Parse JSON
        content = content.strip()
        content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                parsed = {}

        # Ensure required fields
        result = {
            "verdict": parsed.get("verdict", verdict),
            "confidence": parsed.get("confidence", confidence),
            "summary": parsed.get("summary", f"Kết quả kiểm chứng là {verdict}."),
            "detailed_explanation": parsed.get("detailed_explanation", reasoning),
            "citations": parsed.get("citations", []),
            "label": parsed.get("verdict", verdict)
        }

        logger.info("[AGENT] Successfully generated explanation report")
        return result

    except Exception as e:
        logger.error(f"[AGENT] Explanation generation failed: {e}")
        return {
            "verdict": verdict,
            "confidence": confidence,
            "summary": f"Lỗi khi tạo báo cáo: {str(e)}",
            "detailed_explanation": reasoning,
            "label": verdict
        }


def run_explainer_agent_sync(
    claim: str,
    verdict_data: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Synchronous wrapper for run_explainer_agent.

    Handles nested asyncio loops properly (works from within asyncio.run).
    """
    import asyncio
    import concurrent.futures

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_explainer_agent(claim, verdict_data, evidence))

    def _run_in_thread():
        return asyncio.run(run_explainer_agent(claim, verdict_data, evidence))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_in_thread).result()
