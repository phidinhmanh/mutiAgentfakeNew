"""
Explainer Agent Tools - Tools for generating explanations of verification results.

Tools used by the Explainer ReAct Agent:
- Summarize Tool: Summarize the verification process
- Generate Explanation Tool: Create natural language explanation
- Cite Evidence Tool: Format evidence citations
- Create Report Tool: Generate comprehensive report

Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import json
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool

from trust_agents.llm.llm_helpers import call_llm

load_dotenv()
logger = logging.getLogger("TRUST_agents.agents.explainer_tools")
logger.propagate = True


def safe_json_parse(data: str | dict | list | Any, default: Any = None) -> Any:
    """
    Safely parse JSON data that might be a string, dict, list, or other type.
    """
    if data is None:
        return default

    if isinstance(data, (dict, list)):
        return data

    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return default if default is not None else data

    return default if default is not None else str(data)


@tool()
async def summarize_verification_tool(
    claim: str, verdict: str, confidence: float, evidence_summary: str
) -> str:
    """
    Summarize the verification process and key findings.

    Args:
        claim: The original claim
        verdict: The verification verdict (true/false/uncertain)
        confidence: Confidence score (0.0-1.0)
        evidence_summary: Summary of evidence used

    Returns:
        JSON string with verification summary
    """
    logger.info("[DEBUG] summarize_verification_tool called")

    try:
        if not isinstance(confidence, (int, float)):
            try:
                confidence = float(confidence)
            except:  # noqa: E722
                confidence = 0.5

        prompt = f"""Summarize this fact-checking verification in 2-3 clear sentences.

Claim: {claim}
Verdict: {verdict}
Confidence: {confidence:.1%}
Evidence: {evidence_summary}

Create a concise summary that:
1. States the claim outcome clearly
2. Mentions the key evidence used
3. Explains the confidence level
"""

        summary = call_llm(
            prompt,
            system_prompt="You are a fact-checking expert. Write clear, concise summaries.",
            max_tokens=300,
        )

        result = {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "summary": summary,
            "evidence_count": len(evidence_summary.split("Evidence")) - 1
            if evidence_summary
            else 0,
        }

        logger.info("summarize_verification_tool completed")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error in summarize_verification_tool: {e}")
        return json.dumps(
            {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "summary": f"Verification result: {verdict} with {confidence:.1%} confidence",
                "error": str(e),
            }
        )


@tool()
async def generate_explanation_tool(
    claim: str, verdict_data: str, evidence_list: str
) -> str:
    """
    Generate detailed natural language explanation with evidence citations.

    Args:
        claim: The original claim
        verdict_data: JSON string with verdict data
        evidence_list: JSON string with list of evidence passages

    Returns:
        JSON string with detailed explanation
    """
    logger.info("[DEBUG] generate_explanation_tool called")

    try:
        verdict = safe_json_parse(
            verdict_data, {"verdict": "uncertain", "confidence": 0.5}
        )
        evidence = safe_json_parse(evidence_list, [])

        verdict_str = verdict.get("verdict", "uncertain")
        confidence = verdict.get("confidence", 0.5)
        reasoning = verdict.get("reasoning", "Based on available evidence.")

        # Format evidence for prompt
        evidence_texts = []
        for i, ev in enumerate(evidence[:3]):
            if isinstance(ev, dict):
                evidence_texts.append(f"[{i + 1}] {ev.get('text', str(ev))[:200]}...")
            else:
                evidence_texts.append(f"[{i + 1}] {str(ev)[:200]}...")

        evidence_str = "\n\n".join(evidence_texts)

        prompt = f"""Generate a detailed explanation of this fact-check result.

Claim: {claim}
Verdict: {verdict_str} (confidence: {confidence:.1%})

Reasoning: {reasoning}

Supporting Evidence:
{evidence_str}

Write a clear explanation that:
1. States whether the claim is TRUE, FALSE, or UNCERTAIN
2. Explains what evidence supports this conclusion
3. Addresses any contradictions or gaps in evidence
4. Provides context about the topic
"""

        explanation = call_llm(
            prompt,
            system_prompt="You are a fact-checking expert. Write detailed, objective explanations.",
            max_tokens=600,
        )

        result = {
            "claim": claim,
            "verdict": verdict_str,
            "confidence": confidence,
            "explanation": explanation,
            "evidence_used": len(evidence),
        }

        logger.info("generate_explanation_tool completed")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error in generate_explanation_tool: {e}")
        return json.dumps(
            {
                "claim": claim,
                "explanation": f"Unable to generate detailed explanation: {str(e)}",
                "error": str(e),
            }
        )


@tool()
async def cite_evidence_tool(evidence_list: str) -> str:
    """
    Format evidence citations according to standard format.

    Args:
        evidence_list: JSON string with list of evidence passages

    Returns:
        JSON string with formatted citations
    """
    logger.info("[DEBUG] cite_evidence_tool called")

    try:
        evidence = safe_json_parse(evidence_list, [])

        citations = []
        for i, ev in enumerate(evidence[:5]):
            if isinstance(ev, dict):
                source = ev.get("source", ev.get("url", "Unknown source"))
                text = ev.get("text", str(ev))[:150]
                citations.append(f"[{i + 1}] {text}... (Source: {source})")
            else:
                citations.append(f"[{i + 1}] {str(ev)[:150]}...")

        result = {"citations": citations, "count": len(citations)}

        logger.info(f"cite_evidence_tool completed: {len(citations)} citations")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error in cite_evidence_tool: {e}")
        return json.dumps({"citations": [], "error": str(e)})


@tool()
async def create_report_tool(
    claim: str,
    verdict: str,
    confidence: float,
    summary: str,
    explanation: str,
    citations: str,
) -> str:
    """
    Compile comprehensive fact-check report from all components.

    Args:
        claim: The original claim
        verdict: The verification verdict
        confidence: Confidence score
        summary: Verification summary
        explanation: Detailed explanation
        citations: Formatted evidence citations

    Returns:
        JSON string with complete fact-check report
    """
    logger.info("[DEBUG] create_report_tool called")

    try:
        parsed_citations = safe_json_parse(citations, {"citations": [], "count": 0})
        if not isinstance(parsed_citations, dict):
            parsed_citations = {"citations": [], "count": 0}

        verdict_str = verdict if isinstance(verdict, str) else str(verdict)
        summary_str = summary if isinstance(summary, str) else str(summary)
        explanation_str = (
            explanation if isinstance(explanation, str) else str(explanation)
        )
        confidence_value = (
            float(confidence) if isinstance(confidence, (int, float)) else 0.5
        )

        report = {
            "claim": claim,
            "verdict": verdict_str,
            "confidence": confidence_value,
            "label": verdict_str,
            "summary": summary_str,
            "explanation": explanation_str,
            "citations": parsed_citations.get("citations", []),
            "evidence_count": parsed_citations.get("count", 0),
        }

        logger.info("create_report_tool completed")
        return json.dumps(report)

    except Exception as e:
        logger.error(f"Error in create_report_tool: {e}")
        return json.dumps(
            {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "label": verdict,
                "summary": summary,
                "explanation": explanation,
                "error": str(e),
            }
        )
