"""
Verifier Agent - ReAct Agent for claim verification.

Uses evidence passages to verify claims and generate verdicts.
Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import json
import logging
from typing import Any

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

from trust_agents.agents.verifier_tools import (
    aggregate_evidence_tool,
    compare_claim_evidence_tool,
    confidence_calibration_tool,
    generate_verdict_tool,
)
from trust_agents.llm.factory import create_chat_model
from trust_agents.parsing import extract_last_message_text, parse_dict_payload

load_dotenv()
logger = logging.getLogger("Verifier.Agent")


async def run_verifier_agent(
    claim: str, evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    """Verify a claim against evidence passages."""
    model = create_chat_model()

    logger.info("[AGENT] Verifier Agent initialized")

    evidence_summary = "\n\n".join(
        [
            (
                f"Evidence {index + 1} "
                f"(score: {item.get('hybrid_score', item.get('score', 0.5)):.3f}):\n"
                f"{item.get('text', str(item))[:300]}..."
            )
            for index, item in enumerate(evidence[:5])
        ]
    )
    evidence_json = json.dumps(evidence[:5], ensure_ascii=False)

    agent_prompt = f"""
You are a Verifier agent. Your task is to verify the claim against the provided evidence.

Claim: {claim}

Evidence Passages:
{evidence_summary}

You have access to these tools:
- compare_claim_evidence_tool: Compare claim against individual evidence passages
- aggregate_evidence_tool: Aggregate multiple evidence assessments
- generate_verdict_tool: Generate final verdict based on aggregated assessment
- confidence_calibration_tool: Calibrate confidence based on evidence quality

Your PROCESS:
1. Use aggregate_evidence_tool with the evidence JSON: {evidence_json[:500]}...
2. Use generate_verdict_tool with the aggregated assessment
3. Return the final verdict

After verification, return JSON: {{"verdict": "true|false|uncertain", "confidence": 0.0-1.0, "reasoning": "explanation"}}.
""".strip()  # noqa: E501

    tools = [
        compare_claim_evidence_tool,
        aggregate_evidence_tool,
        generate_verdict_tool,
        confidence_calibration_tool,
    ]
    logger.info("[AGENT] Loaded %d tools.", len(tools))

    agent = create_react_agent(model, tools)
    logger.info("[AGENT] LangGraph agent initialized. Invoking agent...")

    result: dict[str, Any] = await agent.ainvoke(
        {"messages": [{"role": "user", "content": agent_prompt}]}
    )

    msgs = result.get("messages", [])
    logger.debug("[AGENT] Received %d messages", len(msgs))
    for index, msg in enumerate(msgs):
        content = extract_last_message_text([msg])
        logger.debug(
            "Message %d | %s | len=%d\n%s\n",
            index,
            type(msg).__name__,
            len(content),
            content,
        )

    logger.info("[AGENT] Agent execution completed. Processing results...")
    parsed = parse_dict_payload(extract_last_message_text(msgs))
    if parsed and "verdict" in parsed:
        logger.info("[AGENT] Successfully extracted verdict: %s", parsed.get("verdict"))
        return parsed

    for msg in reversed(msgs):
        message_text = extract_last_message_text([msg])
        parsed = parse_dict_payload(message_text)
        if parsed and "verdict" in parsed:
            logger.info(
                "[AGENT] Recovered verdict from intermediate message: %s",
                parsed.get("verdict"),
            )
            return parsed
        if parsed and "overall_verdict" in parsed:
            overall_verdict = str(parsed.get("overall_verdict", "insufficient")).lower()
            verdict_map = {
                "supported": "true",
                "contradicted": "false",
                "insufficient": "uncertain",
                "error": "uncertain",
            }
            verdict = verdict_map.get(overall_verdict, "uncertain")
            recovered = {
                "claim": claim,
                "verdict": verdict,
                "confidence": float(parsed.get("confidence", 0.3)),
                "label": verdict,
                "reasoning": parsed.get(
                    "reasoning", "Recovered from aggregated assessment"
                ),
                "evidence_summary": {
                    "overall_verdict": overall_verdict,
                    "supporting_count": parsed.get("supporting_count", 0),
                    "contradicting_count": parsed.get("contradicting_count", 0),
                    "key_points": parsed.get("key_points", []),
                    "conflicts": parsed.get("conflicts", []),
                },
            }
            logger.info(
                "[AGENT] Recovered verdict from aggregated assessment: %s",
                recovered.get("verdict"),
            )
            return recovered

    logger.warning("[AGENT] No verdict found, returning uncertain")
    return {
        "claim": claim,
        "verdict": "uncertain",
        "confidence": 0.3,
        "label": "uncertain",
        "reasoning": "Unable to determine verdict from available evidence",
    }


def run_verifier_agent_sync(
    claim: str, evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    """Synchronous wrapper for run_verifier_agent."""
    import asyncio

    return asyncio.run(run_verifier_agent(claim, evidence))
