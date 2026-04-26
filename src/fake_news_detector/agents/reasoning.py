"""Agent 3: Reasoning Agent - NVIDIA NIM + citation validation."""

import json
import logging
import re
from collections.abc import Generator
from typing import Any

from fake_news_detector.utils.citation_checker import validate_citation

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là chuyên gia xác thực tin tức Việt Nam.

CHỈ sử dụng thông tin trong phần Evidence để giải thích.
NẾU Evidence không chứa thông tin liên quan, trả về 'Không đủ bằng chứng'.

YÊU CẦU trả về JSON với citation chính xác:
{
  "verdict": "REAL|FAKE|UNVERIFIABLE",
  "confidence": 0.0-1.0,
  "reasoning": "Giải thích dựa trên evidence",
  "citations": [
    {"evidence_id": 0, "quote_text": "trích dẫn chính xác từ evidence"}
  ]
}

QUAN TRỌNG: Quote text phải tồn tại nguyên trong Evidence."""


class ReasoningAgent:
    """Agent 3: Reasoning with constraint prompting and citation validation."""

    def __init__(self, llm_client: Any) -> None:
        """Initialize reasoning agent.

        Args:
            llm_client: NVIDIA NIM API client
        """
        self.llm = llm_client

    def analyze(self, claim: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze claim with evidence and return verdict.

        Args:
            claim: Claim to verify
            evidence: List of evidence items

        Returns:
            Verdict with reasoning and citations
        """
        if not evidence:
            return self._unverifiable_result("Không có bằng chứng")

        evidence_text = self._format_evidence(evidence)

        prompt = f"""Claim: {claim}

Evidence:
{evidence_text}

{SYSTEM_PROMPT}
"""

        try:
            response = self.llm.invoke(prompt)
            result = self._parse_response(response)

            validated_result = self._validate_citations(result, evidence)
            return validated_result
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return self._unverifiable_result(f"Lỗi xử lý: {str(e)}")

    def stream_analysis(
        self, claim: str, evidence: list[dict[str, Any]]
    ) -> Generator[str, None, None]:
        """Stream analysis response for UI.

        Args:
            claim: Claim to verify
            evidence: List of evidence items

        Yields:
            Text chunks of the response
        """
        if not evidence:
            yield json.dumps(self._unverifiable_result("Không có bằng chứng"))
            return

        evidence_text = self._format_evidence(evidence)

        prompt = f"""Claim: {claim}

Evidence:
{evidence_text}

{SYSTEM_PROMPT}
"""

        try:
            yield from self.llm.stream(prompt)
        except Exception as e:
            logger.error(f"Stream reasoning failed: {e}")
            yield json.dumps(self._unverifiable_result(f"Lỗi xử lý: {str(e)}"))

    def _format_evidence(self, evidence: list[dict[str, Any]]) -> str:
        """Format evidence for prompt."""
        formatted = []
        for i, ev in enumerate(evidence):
            formatted.append(
                f"[{i}] {ev.get('content', '')} (Source: {ev.get('source', 'unknown')})"
            )
        return "\n".join(formatted)

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM JSON response."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        return {
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": response[:500],
            "citations": [],
        }

    def _validate_citations(
        self, result: dict[str, Any], evidence: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Validate that citations actually exist in evidence."""
        validated_citations = []

        for citation in result.get("citations", []):
            evidence_id = citation.get("evidence_id", -1)
            quote_text = citation.get("quote_text", "")

            if evidence_id < len(evidence):
                if validate_citation(quote_text, evidence):
                    validated_citations.append(citation)
                else:
                    logger.warning(
                        f"Citation validation failed: '{quote_text[:50]}...'"
                    )
            else:
                logger.warning(f"Invalid evidence_id: {evidence_id}")

        return {
            **result,
            "citations": validated_citations,
            "citations_valid": len(validated_citations)
            == len(result.get("citations", [])),
        }

    def _unverifiable_result(self, reason: str) -> dict[str, Any]:
        """Return unverifiable result."""
        return {
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": reason,
            "citations": [],
            "citations_valid": True,
        }


def aggregate_verdicts(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple verdicts into final decision.

    Args:
        verdicts: List of individual verdicts

    Returns:
        Aggregated verdict
    """
    if not verdicts:
        return {
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": "Không có kết quả",
        }

    real_count = sum(1 for v in verdicts if v.get("verdict") == "REAL")
    fake_count = sum(1 for v in verdicts if v.get("verdict") == "FAKE")
    total = len(verdicts)

    if fake_count > real_count:
        verdict = "FAKE"
    elif real_count > fake_count:
        verdict = "REAL"
    else:
        verdict = "UNVERIFIABLE"

    confidences = [v.get("confidence", 0.0) for v in verdicts]
    avg_confidence = sum(confidences) / len(confidences)

    all_reasoning = " ".join(v.get("reasoning", "") for v in verdicts)

    return {
        "verdict": verdict,
        "confidence": avg_confidence,
        "reasoning": all_reasoning[:1000],
        "stats": {
            "real_count": real_count,
            "fake_count": fake_count,
            "total": total,
        },
    }
