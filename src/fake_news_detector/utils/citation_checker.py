"""Compatibility wrapper for trust_agents.utils.citation_checker."""

from trust_agents.utils.citation_checker import (
    _find_matching_evidence,
    _fuzzy_match,
    _normalize_text,
    extract_citations_from_text,
    validate_citation,
    verify_citation_evidence_pairs,
)

__all__ = [
    "_find_matching_evidence",
    "_fuzzy_match",
    "_normalize_text",
    "extract_citations_from_text",
    "validate_citation",
    "verify_citation_evidence_pairs",
]
