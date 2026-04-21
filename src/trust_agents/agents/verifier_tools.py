# -*- coding: utf-8 -*-

"""
Verifier Agent Tools - Tools for claim verification against evidence.

Tools used by the Verifier ReAct Agent:
- Compare Tool: Compare claim against evidence passages
- Score Tool: Generate verification score
- Verdict Tool: Determine final verdict (true/false/uncertain)

Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import os
import json
import logging
from typing import List, Dict

from langchain_core.tools import tool
from dotenv import load_dotenv
from trust_agents.llm.llm_helpers import call_llm, call_llm_json

load_dotenv()
logger = logging.getLogger("TRUST_agents.agents.verifier_tools")
logger.propagate = True


@tool()
async def compare_claim_evidence_tool(claim: str, evidence_text: str) -> str:
    """
    Compare a claim against evidence text to assess consistency.

    Args:
        claim: The claim to verify
        evidence_text: The evidence text to compare against

    Returns:
        JSON string with comparison result and reasoning
    """
    logger.info(f"[DEBUG] compare_claim_evidence_tool called")

    try:
        prompt = f"""Compare the following claim against the evidence and assess consistency.

Claim: {claim}

Evidence: {evidence_text}

Analyze:
1. Does the evidence support the claim?
2. Does the evidence contradict the claim?
3. Is the evidence insufficient to determine?

Return ONLY valid JSON:
{{
"consistency": "supports|contradicts|insufficient",
"confidence": 0.0-1.0,
"key_points": ["point1", "point2"],
"reasoning": "brief explanation"
}}"""

        result = call_llm_json(
            prompt,
            system_prompt="You are a fact verification expert. Analyze claims against evidence objectively.",
            max_tokens=500,
        )

        logger.info(f"compare_claim_evidence_tool completed: {result.get('consistency')}")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error comparing claim and evidence: {e}")
        return json.dumps({
            "consistency": "error",
            "confidence": 0.0,
            "key_points": [],
            "reasoning": str(e),
            "error": str(e)
        })


@tool()
async def aggregate_evidence_tool(claim: str, evidence_list: str) -> str:
    """
    Aggregate multiple evidence passages to form an overall assessment.

    Args:
        claim: The claim being verified
        evidence_list: JSON string containing list of evidence passages with scores

    Returns:
        JSON string with aggregated assessment
    """
    logger.info(f"[DEBUG] aggregate_evidence_tool called")

    try:
        # Parse evidence list
        try:
            evidence_data = json.loads(evidence_list)
            if not isinstance(evidence_data, list):
                evidence_data = [evidence_data]
        except:
            evidence_data = [{"text": evidence_list, "score": 0.5}]

        # Format evidence for prompt
        evidence_summary = "\n\n".join([
            f"Evidence {i+1} (relevance: {ev.get('hybrid_score', ev.get('score', 0.5)):.3f}):\n{ev.get('text', str(ev))}"
            for i, ev in enumerate(evidence_data[:5])  # Top 5 pieces
        ])

        prompt = f"""You are a fact-checker. Aggregate the following evidence to verify the claim.

Claim: {claim}

Evidence Passages:
{evidence_summary}

INSTRUCTIONS:
- If the MAJORITY of evidence supports the claim → verdict: "supported"
- If the MAJORITY of evidence contradicts the claim → verdict: "contradicted"
- ONLY use "insufficient" if evidence is truly ambiguous or off-topic
- Be decisive - favor "supported" or "contradicted" when you have ANY relevant evidence
- Confidence should reflect strength of evidence (0.3-0.5 = weak, 0.5-0.7 = moderate, 0.7-1.0 = strong)

Return ONLY valid JSON:
{{
"overall_verdict": "supported|contradicted|insufficient",
"confidence": 0.3-1.0,
"supporting_count": 0,
"contradicting_count": 0,
"key_points": ["point1", "point2"],
"conflicts": ["conflict1"],
"reasoning": "explanation"
}}

IMPORTANT: Make a decision (supported/contradicted) unless evidence is truly ambiguous."""

        result = call_llm_json(
            prompt,
            system_prompt="You are a decisive fact verification expert. Make clear judgments based on available evidence.",
            max_tokens=600,
        )

        # Boost confidence slightly if we have multiple pieces of evidence
        if len(evidence_data) >= 3:
            original_conf = result.get("confidence", 0.5)
            result["confidence"] = min(original_conf * 1.15, 0.95)

        logger.info(f"aggregate_evidence_tool completed: {result.get('overall_verdict')} (confidence: {result.get('confidence'):.3f})")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error aggregating evidence: {e}")
        return json.dumps({
            "overall_verdict": "insufficient",
            "confidence": 0.3,
            "supporting_count": 0,
            "contradicting_count": 0,
            "key_points": [],
            "conflicts": [],
            "reasoning": str(e),
            "error": str(e)
        })


@tool()
async def generate_verdict_tool(claim: str, aggregated_assessment: str) -> str:
    """
    Generate final verdict based on aggregated evidence assessment.

    Args:
        claim: The original claim
        aggregated_assessment: JSON string with aggregated evidence assessment

    Returns:
        JSON string with final verdict
    """
    logger.info(f"[DEBUG] generate_verdict_tool called")

    try:
        # Parse assessment
        try:
            assessment = json.loads(aggregated_assessment)
        except:
            logger.warning("Failed to parse assessment, using defaults")
            assessment = {"overall_verdict": "insufficient", "confidence": 0.3}

        # Generate verdict based on assessment
        verdict_map = {
            "supported": "true",
            "contradicted": "false",
            "insufficient": "uncertain",
            "error": "uncertain"
        }

        overall_verdict = assessment.get("overall_verdict", "insufficient")
        verdict = verdict_map.get(overall_verdict, "uncertain")
        confidence = assessment.get("confidence", 0.3)

        # Lower threshold from 0.4 to 0.25
        # Only force "uncertain" if confidence is VERY low
        if confidence < 0.25:
            logger.warning(f"Low confidence ({confidence:.3f}), forcing uncertain")
            verdict = "uncertain"
            confidence = 0.25

        result = {
            "claim": claim,
            "verdict": verdict,
            "confidence": float(confidence),
            "label": verdict,
            "evidence_summary": {
                "overall_verdict": assessment.get("overall_verdict", "insufficient"),
                "supporting_count": assessment.get("supporting_count", 0),
                "contradicting_count": assessment.get("contradicting_count", 0),
                "key_points": assessment.get("key_points", []),
                "conflicts": assessment.get("conflicts", [])
            },
            "reasoning": assessment.get("reasoning", "Based on available evidence")
        }

        logger.info(f"generate_verdict_tool completed: {verdict} (confidence: {confidence:.3f})")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error generating verdict: {e}")
        return json.dumps({
            "claim": claim,
            "verdict": "uncertain",
            "confidence": 0.3,
            "label": "uncertain",
            "reasoning": f"Error: {str(e)}",
            "error": str(e)
        })


@tool()
async def confidence_calibration_tool(verdict: str, evidence_quality: str) -> str:
    """
    Calibrate confidence score based on evidence quality and consistency.

    Args:
        verdict: The current verdict (true/false/uncertain)
        evidence_quality: JSON string describing evidence quality metrics

    Returns:
        JSON string with calibrated confidence
    """
    logger.info(f"[DEBUG] confidence_calibration_tool called")

    try:
        # Parse quality metrics
        try:
            quality = json.loads(evidence_quality)
        except:
            quality = {"relevance": 0.5, "consistency": 0.5, "quantity": 1}

        # Calibration factors
        relevance = quality.get("relevance", 0.5)
        consistency = quality.get("consistency", 0.5)
        quantity = min(quality.get("quantity", 1) / 5.0, 1.0)

        base_confidence = quality.get("base_confidence", 0.5)

        # Apply calibration
        calibrated = base_confidence * (
            0.4 * relevance +
            0.4 * consistency +
            0.2 * quantity
        )

        if verdict == "uncertain":
            calibrated *= 0.9

        calibrated = max(calibrated, 0.25)

        result = {
            "original_confidence": base_confidence,
            "calibrated_confidence": round(calibrated, 3),
            "factors": {
                "relevance": relevance,
                "consistency": consistency,
                "quantity": quantity
            },
            "verdict": verdict
        }

        logger.info(f"confidence_calibration_tool completed: {calibrated:.3f}")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error calibrating confidence: {e}")
        return json.dumps({
            "original_confidence": 0.5,
            "calibrated_confidence": 0.35,
            "error": str(e)
        })