"""Citation checker to validate quotes against evidence."""

from typing import Any


def validate_citation(quote_text: str, evidence_list: list[dict[str, Any]]) -> bool:
    """Validate that a quoted text exists in evidence.

    Args:
        quote_text: The quoted text to validate
        evidence_list: List of evidence documents

    Returns:
        True if citation is valid
    """
    if not quote_text or not evidence_list:
        return False

    quote_normalized = _normalize_text(quote_text)

    for evidence in evidence_list:
        content = evidence.get("content", "")
        content_normalized = _normalize_text(content)

        if quote_normalized in content_normalized:
            return True

        if _fuzzy_match(quote_normalized, content_normalized) > 0.8:
            return True

    return False


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".,!?;:'\"-()[]{}")
    return text


def _fuzzy_match(text1: str, text2: str, min_length: int = 20) -> float:
    """Calculate fuzzy match score between two texts.

    Args:
        text1: First text
        text2: Second text
        min_length: Minimum length for meaningful comparison

    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0

    if len(text1) < min_length or len(text2) < min_length:
        return 0.0

    words1 = set(text1.split())
    words2 = set(text2.split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def extract_citations_from_text(text: str) -> list[str]:
    """Extract quoted text that looks like citations.

    Args:
        text: Input text

    Returns:
        List of potential citations
    """
    import re

    # Match double quotes, single quotes, or French quotes with content >= 20 chars
    pattern = r'"([^"]{20,})"|\'([^\']{20,})\'|«([^»]{20,})»'
    matches = re.findall(pattern, text)

    citations = []
    for match in matches:
        for group in match:
            if group:
                citations.append(group)

    return citations


def verify_citation_evidence_pairs(
    citations: list[dict[str, str]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Verify citation-evidence pairs.

    Args:
        citations: List of {"quote": ..., "source": ...}
        evidence: List of evidence documents

    Returns:
        List of verification results
    """
    results = []

    for citation in citations:
        quote = citation.get("quote", "")
        is_valid = validate_citation(quote, evidence)

        results.append(
            {
                "quote": quote,
                "is_valid": is_valid,
                "matched_evidence": _find_matching_evidence(quote, evidence),
            }
        )

    return results


def _find_matching_evidence(quote: str, evidence: list[dict[str, Any]]) -> list[int]:
    """Find indices of evidence containing the quote."""
    indices = []
    quote_normalized = _normalize_text(quote)

    for i, ev in enumerate(evidence):
        content_normalized = _normalize_text(ev.get("content", ""))
        if quote_normalized in content_normalized:
            indices.append(i)

    return indices
