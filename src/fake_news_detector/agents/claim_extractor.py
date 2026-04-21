"""Agent 1: Claim Extractor - Extract and verify claims."""
import json
import logging
import re
from typing import Any

from fake_news_detector.data.preprocessing import split_sentences

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là chuyên gia trích xuất thông tin từ bài viết tiếng Việt.
Nhiệm vụ của bạn:
1. Trích xuất các claims (tuyên bố) từ bài viết
2. Phân loại mỗi claim là FACT (sự thật) hay OPINION (ý kiến)

Quy tắc:
- FACT: Khẳng định có thể kiểm chứng bằng dữ liệu, sự kiện
- OPINION: Ý kiến cá nhân, suy luận chủ quan

Trả về JSON với format:
{
  "claims": [
    {"text": "nội dung claim", "type": "FACT|OPINION", "verifiable": true/false}
  ]
}
"""


def extract_claims(article: str) -> list[dict[str, Any]]:
    """Extract claims from article text.

    Args:
        article: Input article text

    Returns:
        List of extracted claims
    """
    if not article or not article.strip():
        return []

    sentences = split_sentences(article)
    claims = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue

        claim_type = _classify_claim(sent)
        claims.append({
            "text": sent,
            "type": claim_type,
            "verifiable": claim_type == "FACT",
        })

    logger.info(f"Extracted {len(claims)} claims from article")
    return claims


def _classify_claim(sentence: str) -> str:
    """Classify if a sentence is a FACT or OPINION.

    Args:
        sentence: Input sentence

    Returns:
        "FACT" or "OPINION"
    """
    opinion_indicators = {
        "tôi nghĩ", "tôi tin", "theo tôi", "có lẽ", "có thể",
        "chắc chắn", "đảm bảo", "tuyệt đối", "luôn luôn",
        "không bao giờ", "phải", "nên", "không nên",
        "đẹp", "xấu", "tốt", "hay", "dở", "buồn", "vui",
    }
    fact_indicators = {
        "theo", "cho biết", "phát biểu", "nghiên cứu", "thống kê",
        "số liệu", "báo cáo", "xác nhận", "công bố", "điều tra",
        "ngày", "tháng", "năm", "lúc", "giờ", "phút",
        "%", "triệu", "nghìn", "tỷ",
    }

    sentence_lower = sentence.lower()

    opinion_count = sum(1 for ind in opinion_indicators if ind in sentence_lower)
    fact_count = sum(1 for ind in fact_indicators if ind in sentence_lower)

    if fact_count > opinion_count:
        return "FACT"
    elif opinion_count > fact_count:
        return "OPINION"
    else:
        return "FACT" if any(c.isdigit() for c in sentence) else "OPINION"


def verify_facts_with_llm(claims: list[dict[str, Any]], llm: Any) -> list[dict[str, Any]]:
    """Verify facts using LLM.

    Args:
        claims: List of claims to verify
        llm: LLM client for verification

    Returns:
        Claims with verification status
    """
    verified_claims = []

    for claim in claims:
        if claim["type"] != "FACT":
            claim["verified"] = None
            verified_claims.append(claim)
            continue

        try:
            prompt = f"""Kiểm tra claim sau là FACT hay không có cơ sở:
Claim: {claim['text']}

Trả lời JSON: {{"is_verified": true/false, "reason": "giải thích ngắn"}}
"""
            response = llm.invoke(prompt)
            result = _parse_llm_response(response)

            claim["verified"] = result.get("is_verified", False)
            claim["verify_reason"] = result.get("reason", "")
            verified_claims.append(claim)
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            claim["verified"] = None
            verified_claims.append(claim)

    return verified_claims


def _parse_llm_response(response: str) -> dict[str, Any]:
    """Parse LLM JSON response."""
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {}


def filter_verifiable_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter claims to only verifiable facts.

    Args:
        claims: List of all claims

    Returns:
        List of verifiable fact claims
    """
    return [
        c for c in claims
        if c.get("type") == "FACT" and c.get("verifiable", False)
    ]