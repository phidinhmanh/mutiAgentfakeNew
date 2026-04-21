"""Agent 2: Evidence Retriever - Hybrid RAG for evidence gathering."""
import logging
from typing import Any

from fake_news_detector.rag.retriever import retrieve_evidence

logger = logging.getLogger(__name__)


def retrieve_evidence_for_claims(
    claims: list[dict[str, Any]],
    use_web_search: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve evidence for multiple claims.

    Args:
        claims: List of claims to find evidence for
        use_web_search: Whether to use web search fallback

    Returns:
        List of claims with retrieved evidence
    """
    results = []

    for claim in claims:
        claim_text = claim.get("text", "")
        if not claim_text:
            results.append({**claim, "evidence": []})
            continue

        evidence = retrieve_evidence(claim_text, use_web_search=use_web_search)
        logger.info(f"Retrieved {len(evidence)} evidence items for claim: {claim_text[:50]}...")

        results.append({
            **claim,
            "evidence": evidence,
            "num_evidence": len(evidence),
        })

    return results


def merge_evidence_from_multiple_claims(
    claims_with_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge evidence from multiple claims to avoid duplication.

    Args:
        claims_with_evidence: List of claims with their evidence

    Returns:
        Merged list of unique evidence items
    """
    seen_urls = set()
    merged_evidence = []

    for claim_data in claims_with_evidence:
        for evidence in claim_data.get("evidence", []):
            url = evidence.get("url", "")
            content_preview = evidence.get("content", "")[:100]

            if url and url in seen_urls:
                continue
            if content_preview in str(seen_urls):
                continue

            seen_urls.add(url or content_preview)
            merged_evidence.append({
                **evidence,
                "from_claim": claim_data.get("text", "")[:100],
            })

    merged_evidence.sort(key=lambda x: x.get("score", 0), reverse=True)
    return merged_evidence


def get_weak_claims(
    claims_with_evidence: list[dict[str, Any]],
    min_evidence_count: int = 1,
) -> list[dict[str, Any]]:
    """Identify claims with insufficient evidence.

    Args:
        claims_with_evidence: Claims with evidence retrieved
        min_evidence_count: Minimum evidence items needed

    Returns:
        List of claims with weak evidence
    """
    return [
        c for c in claims_with_evidence
        if c.get("num_evidence", 0) < min_evidence_count
    ]


def enrich_evidence_with_context(
    evidence: list[dict[str, Any]],
    claim: str,
) -> list[dict[str, Any]]:
    """Enrich evidence items with relevance context.

    Args:
        evidence: List of evidence items
        claim: Original claim for context

    Returns:
        Enriched evidence items
    """
    enriched = []

    for item in evidence:
        content = item.get("content", "")
        score = item.get("score", 0)

        relevance_keywords = _find_relevant_keywords(claim, content)
        context_overlap = _calculate_context_overlap(claim, content)

        enriched.append({
            **item,
            "relevance_keywords": relevance_keywords,
            "context_overlap": context_overlap,
            "final_score": (score + context_overlap) / 2,
        })

    enriched.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return enriched


def _find_relevant_keywords(claim: str, content: str) -> list[str]:
    """Find keywords present in both claim and content."""
    claim_words = set(claim.lower().split())
    content_words = set(content.lower().split())
    overlap = claim_words & content_words
    return list(overlap)[:10]


def _calculate_context_overlap(claim: str, content: str) -> float:
    """Calculate how much context overlaps between claim and content."""
    claim_lower = claim.lower()
    content_lower = content.lower()

    claim_words = set(claim_lower.split())
    content_words = set(content_lower.split())

    if not claim_words:
        return 0.0

    overlap = claim_words & content_words
    return len(overlap) / len(claim_words)