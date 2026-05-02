"""
Verifier Agent Tools - Tools for claim verification against evidence.

Tools used by the Verifier ReAct Agent:
- Compare Tool: Compare claim against evidence passages
- Score Tool: Generate verification score
- Verdict Tool: Determine final verdict (true/false/uncertain)

Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import json
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool

from trust_agents.llm.llm_helpers import call_llm_json

load_dotenv()
logger = logging.getLogger("TRUST_agents.agents.verifier_tools")
logger.propagate = True


# -------------------------------------------------------------------
# Deterministic fallback helpers — used when LLM JSON parsing fails.
# -------------------------------------------------------------------


def _try_parse_verdict_json(text: str) -> dict[str, Any] | None:
    """Try extracting and parsing a partial JSON verdict from truncated LLM output."""
    import re

    if not text.strip():
        return None

    # Try to find and parse a partial verdict from what the model already emitted.
    patterns = [
        r'"overall_verdict"\s*:\s*"([^"]+)"',
        r'"consistency"\s*:\s*"([^"]+)"',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            verdict_value = m.group(1).lower()
            verdict_map = {
                "supported": "supported",
                "contradicts": "contradicted",
                "contradicted": "contradicted",
                "insufficient": "insufficient",
            }
            verdict = verdict_map.get(verdict_value, "insufficient")

            conf_m = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
            confidence = float(conf_m.group(1)) if conf_m else 0.5

            key_m = re.findall(r'"key_points"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            key_points: list[str] = []
            if key_m:
                key_points = [
                    k.strip().strip('"') for k in re.findall(r'"([^"]+)"', key_m[0])
                ]

            result: dict[str, Any] = {
                "overall_verdict": verdict,
                "confidence": confidence,
                "supporting_count": 0,
                "contradicting_count": 0,
                "key_points": key_points,
                "conflicts": [],
                "reasoning": f"Fallback parse from partial response: {verdict}",
                "_fallback": True,
            }
            logger.warning(
                "aggregate_evidence_tool: recovered partial verdict '%s' from truncated response",
                verdict,
            )
            return result
    return None


def _heuristic_evidence_verdict(
    claim: str, evidence_list: list[dict[str, Any]]
) -> dict[str, Any]:
    """Deterministic verdict when LLM parsing completely fails."""
    claim_lower = claim.lower()
    claim_words = set(claim_lower.split())

    supporting = 0
    contradicting = 0

    for ev in evidence_list[:5]:
        ev_text = (ev.get("text") or str(ev)).lower()
        ev_words = set(ev_text.split())
        overlap = len(claim_words & ev_words)

        # If significant vocabulary overlap, evidence supports claim
        if overlap >= 2 and len(claim_words) > 0:
            ratio = overlap / len(claim_words)
            if ratio >= 0.3:
                supporting += 1
            else:
                contradicting += 1
        else:
            contradicting += 1

    if supporting > contradicting:
        overall = "supported"
    elif contradicting > supporting:
        overall = "contradicted"
    else:
        overall = "insufficient"

    result: dict[str, Any] = {
        "overall_verdict": overall,
        "confidence": 0.45,
        "supporting_count": supporting,
        "contradicting_count": contradicting,
        "key_points": [],
        "conflicts": [],
        "reasoning": f"Heuristic fallback: supporting={supporting}, contradicting={contradicting}",
        "_fallback": True,
    }
    logger.info(
        "aggregate_evidence_tool: heuristic verdict '%s' (supporting=%d, contradicting=%d)",
        overall,
        supporting,
        contradicting,
    )
    return result


def _heuristic_consistency_verdict(claim: str, evidence_text: str) -> dict[str, Any]:
    """Deterministic consistency verdict when LLM parsing fails."""
    claim_lower = claim.lower()
    ev_lower = evidence_text.lower()

    claim_words = set(claim_lower.split())
    ev_words = set(ev_lower.split())
    overlap = len(claim_words & ev_words)

    if len(claim_words) == 0:
        consistency = "insufficient"
        confidence = 0.3
    else:
        ratio = overlap / len(claim_words)
        if ratio >= 0.25:
            consistency = "supports"
            confidence = min(0.4 + ratio * 0.3, 0.75)
        else:
            consistency = "insufficient"
            confidence = 0.35

    return {
        "consistency": consistency,
        "confidence": confidence,
        "key_points": [],
        "reasoning": f"Heuristic fallback: consistency={consistency}, overlap_ratio={ratio:.2f}"
        if len(claim_words) > 0
        else "No claim words to compare",
        "_fallback": True,
    }


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
    logger.info("[DEBUG] compare_claim_evidence_tool called")

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

        logger.info(
            f"compare_claim_evidence_tool completed: {result.get('consistency')}"
        )
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error comparing claim and evidence: {e}")
        fallback_result = _heuristic_consistency_verdict(claim, evidence_text)
        fallback_result["error"] = str(e)
        return json.dumps(fallback_result)


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
    logger.info("[DEBUG] aggregate_evidence_tool called")

    try:
        # Parse evidence list
        try:
            evidence_data = json.loads(evidence_list)
            if not isinstance(evidence_data, list):
                evidence_data = [evidence_data]
        except:  # noqa: E722
            evidence_data = [{"text": evidence_list, "score": 0.5}]

        # Format evidence for prompt
        evidence_summary = "\n\n".join(
            [
                f"Evidence {i + 1} (relevance: {ev.get('hybrid_score', ev.get('score', 0.5)):.3f}):\n{ev.get('text', str(ev))}"  # noqa: E501
                for i, ev in enumerate(evidence_data[:5])  # Top 5 pieces
            ]
        )

        prompt = f"""You are a professional fact-checker. Aggregate the following evidence to verify the claim.

Claim: {claim}

Evidence Passages:
{evidence_summary}

INSTRUCTIONS:
1. Pay close attention to SPECIFIC DETAILS: names, dates, numbers, and especially the PROCESS/METHOD (the "how").
2. A claim is "supported" if the evidence confirms the core assertion and the specific details are plausible within the confirmed context.
   - For timeframes (e.g., "within 6 hours"), if the evidence shows a process taking a similar or shorter time (e.g., "3 hours"), it is SUPPORTED. Different people may take different amounts of time for the same process.
   - If the claim describes a process as "labor-intensive" or "elaborate" and gives a duration (e.g., "6 hours"), and evidence confirms the process is indeed elaborate, accept the duration as plausible even if other sources show "quick" versions exist.
3. A claim is "contradicted" only if:
   - The evidence directly says the opposite (e.g., "X is NOT Y").
   - The evidence describes a DIFFERENT, INCOMPATIBLE process/method than the one in the claim. 
     * Example: Claim says "transported fully assembled", evidence says "shipped in 32 crates". These are mutually exclusive. Mark as CONTRADICTED.
   - The numbers in the claim are LOGICALLY IMPOSSIBLE or PHYSICALLY INCOMPATIBLE given the evidence.
   - IMPORTANT: Individual variations in speed (3h vs 6h) are NOT contradictions.
4. If the evidence confirms the main topic exists but does NOT mention the specific detail, use your best judgment based on the plausibility of the detail within the confirmed context. UNUSUAL details (like "assembled dinosaur") should be treated with more skepticism than COMMON details.
5. IMPORTANT: Distinguish between similar entities. "Bánh gạo" (rice crackers) is NOT "Bánh cuốn" (rice rolls).
6. Be extremely decisive. Avoid "insufficient" if there is any relevant evidence about the event.
7. Confidence should reflect the strength of alignment:
   - 0.8-1.0: Direct, clear support/contradiction of the specific method.
   - 0.5-0.8: Strong inference or plausible alignment/conflict.
   - 0.1-0.5: Weak evidence or partial match.

Return ONLY valid JSON:
{{
"overall_verdict": "supported|contradicted|insufficient",
"confidence": 0.1-1.0,
"supporting_count": 0,
"contradicting_count": 0,
"key_points": ["point1", "point2"],
"conflicts": ["conflict1"],
"reasoning": "Explain your decision, focusing on why specific details match or conflict."
}}

IMPORTANT: You MUST make a decision (supported/contradicted) if the evidence discusses the same entity or event as the claim."""

        result = call_llm_json(
            prompt,
            system_prompt="You are a decisive, detail-oriented fact-checker. You spot subtle contradictions in claims and prioritize accuracy over broad topic matches.",  # noqa: E501
            max_tokens=1000,
        )

        # Boost confidence slightly if we have multiple pieces of evidence
        if len(evidence_data) >= 3:
            original_conf = result.get("confidence", 0.5)
            result["confidence"] = min(original_conf * 1.15, 0.95)

        logger.info(
            f"aggregate_evidence_tool completed: {result.get('overall_verdict')} (confidence: {result.get('confidence'):.3f})"  # noqa: E501
        )
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error aggregating evidence: {e}")
        partial_result = _try_parse_verdict_json(str(e))
        if partial_result is None:
            partial_result = _heuristic_evidence_verdict(claim, evidence_data)
        partial_result["error"] = str(e)
        return json.dumps(partial_result)


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
    logger.info("[DEBUG] generate_verdict_tool called")

    try:
        # Parse assessment
        try:
            assessment = json.loads(aggregated_assessment)
        except:  # noqa: E722
            logger.warning("Failed to parse assessment, using defaults")
            assessment = {"overall_verdict": "insufficient", "confidence": 0.3}

        # Generate verdict based on assessment
        verdict_map = {
            "supported": "true",
            "contradicted": "false",
            "insufficient": "uncertain",
            "error": "uncertain",
        }

        overall_verdict = assessment.get("overall_verdict", "insufficient")
        verdict = verdict_map.get(overall_verdict, "uncertain")
        confidence = assessment.get("confidence", 0.3)

        # For Vietnamese context, be more decisive
        # Only force "uncertain" if confidence is VERY low (virtually no relevant evidence)
        if confidence < 0.08:
            logger.warning(f"Very low confidence ({confidence:.3f}), forcing uncertain")
            verdict = "uncertain"
            confidence = max(0.08, confidence)
        # If evidence exists (supporting_count + contradicting_count > 0), increase confidence
        # This prevents "uncertain" when there IS evidence available
        elif assessment.get("supporting_count", 0) + assessment.get("contradicting_count", 0) > 0:
            if confidence < 0.25:
                confidence = min(0.25, 0.15 + confidence)
                logger.info(f"Evidence exists but low confidence, boosting to {confidence:.3f}")

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
                "conflicts": assessment.get("conflicts", []),
            },
            "reasoning": assessment.get("reasoning", "Based on available evidence"),
        }

        logger.info(
            f"generate_verdict_tool completed: {verdict} (confidence: {confidence:.3f})"
        )
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error generating verdict: {e}")
        return json.dumps(
            {
                "claim": claim,
                "verdict": "uncertain",
                "confidence": 0.3,
                "label": "uncertain",
                "reasoning": f"Error: {str(e)}",
                "error": str(e),
            }
        )


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
    logger.info("[DEBUG] confidence_calibration_tool called")

    try:
        # Parse quality metrics
        try:
            quality = json.loads(evidence_quality)
        except:  # noqa: E722
            quality = {"relevance": 0.5, "consistency": 0.5, "quantity": 1}

        # Calibration factors
        try:
            relevance = float(quality.get("relevance", 0.5))
        except (ValueError, TypeError):
            relevance = 0.5

        try:
            consistency = float(quality.get("consistency", 0.5))
        except (ValueError, TypeError):
            consistency = 0.5

        try:
            raw_quantity = quality.get("quantity", 1)
            if isinstance(raw_quantity, (list, dict)):
                raw_quantity = len(raw_quantity)
            quantity = min(float(raw_quantity) / 5.0, 1.0)
        except (ValueError, TypeError):
            quantity = 0.2

        try:
            base_confidence = float(quality.get("base_confidence", 0.5))
        except (ValueError, TypeError):
            base_confidence = 0.5

        # Apply calibration
        calibrated = base_confidence * (
            0.4 * relevance + 0.4 * consistency + 0.2 * quantity
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
                "quantity": quantity,
            },
            "verdict": verdict,
        }

        logger.info(f"confidence_calibration_tool completed: {calibrated:.3f}")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error calibrating confidence: {e}")
        return json.dumps(
            {"original_confidence": 0.5, "calibrated_confidence": 0.35, "error": str(e)}
        )
