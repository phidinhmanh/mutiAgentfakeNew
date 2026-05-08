"""Tests for citation checker utilities."""

from __future__ import annotations

from fake_news_detector.utils.citation_checker import (
    _find_matching_evidence,
    _fuzzy_match,
    _normalize_text,
    extract_citations_from_text,
    validate_citation,
    verify_citation_evidence_pairs,
)


class TestNormalizeText:
    """Test _normalize_text function."""

    def test_normalize_lowercase(self) -> None:
        """Text is converted to lowercase."""
        result = _normalize_text("HELLO WORLD")
        assert result == "hello world"

    def test_normalize_whitespace(self) -> None:
        """Multiple whitespace is collapsed."""
        result = _normalize_text("hello   world\n\ttab")
        assert "  " not in result
        assert "\n" not in result

    def test_normalize_punctuation(self) -> None:
        """Punctuation is stripped from start/end only."""
        result = _normalize_text("!hello world,")
        assert result == "hello world"

    def test_normalize_quotes(self) -> None:
        """Quotes are stripped."""
        result = _normalize_text('"hello"')
        assert '"' not in result

    def test_normalize_parentheses(self) -> None:
        """Parentheses are stripped."""
        result = _normalize_text("(hello)")
        assert "(" not in result
        assert ")" not in result


class TestFuzzyMatch:
    """Test _fuzzy_match function."""

    def test_fuzzy_match_identical(self) -> None:
        """Identical texts return 1.0."""
        text = "the quick brown fox jumps over the lazy dog"
        result = _fuzzy_match(text, text)
        assert result == 1.0

    def test_fuzzy_match_empty_text1(self) -> None:
        """Empty first text returns 0.0."""
        result = _fuzzy_match("", "hello world")
        assert result == 0.0

    def test_fuzzy_match_empty_text2(self) -> None:
        """Empty second text returns 0.0."""
        result = _fuzzy_match("hello world", "")
        assert result == 0.0

    def test_fuzzy_match_both_empty(self) -> None:
        """Both empty returns 0.0."""
        result = _fuzzy_match("", "")
        assert result == 0.0

    def test_fuzzy_match_below_min_length(self) -> None:
        """Text below min_length returns 0.0."""
        result = _fuzzy_match("hi", "hello", min_length=20)
        assert result == 0.0

    def test_fuzzy_match_partial_overlap(self) -> None:
        """Partial overlap returns partial score."""
        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick red fox jumps over the lazy cat"
        result = _fuzzy_match(text1, text2)
        assert 0.5 < result < 1.0

    def test_fuzzy_match_no_overlap(self) -> None:
        """No overlap returns 0.0."""
        result = _fuzzy_match(
            "the quick brown fox jumps",
            "completely different text here",
        )
        assert result < 0.2


class TestValidateCitation:
    """Test validate_citation function."""

    def test_validate_empty_quote(self) -> None:
        """Empty quote returns False."""
        evidence = [{"content": "Some content"}]
        result = validate_citation("", evidence)
        assert result is False

    def test_validate_empty_evidence(self) -> None:
        """Empty evidence returns False."""
        result = validate_citation("Some text", [])
        assert result is False

    def test_validate_exact_match(self) -> None:
        """Exact match returns True."""
        evidence = [{"content": "Vietnam GDP grew 8% in 2023"}]
        result = validate_citation("Vietnam GDP grew 8%", evidence)
        assert result is True

    def test_validate_case_insensitive(self) -> None:
        """Case differences are ignored."""
        evidence = [{"content": "Vietnam GDP grew 8%"}]
        result = validate_citation("vietnam gdp grew 8%", evidence)
        assert result is True

    def test_validate_substring_match(self) -> None:
        """Quote substring in evidence returns True."""
        evidence = [{"content": "According to World Bank, Vietnam GDP grew 8% in 2023"}]
        result = validate_citation("Vietnam GDP grew 8%", evidence)
        assert result is True

    def test_validate_no_match(self) -> None:
        """Non-matching quote returns False."""
        evidence = [{"content": "Completely different information here"}]
        result = validate_citation("Some unrelated claim", evidence)
        assert result is False

    def test_validate_fuzzy_match_above_threshold(self) -> None:
        """High similarity fuzzy match returns True."""
        evidence = [{"content": "Vietnam GDP grew 8% in 2023, fastest in ASEAN"}]
        result = validate_citation("Vietnam GDP grew 8% in 2023", evidence)
        assert result is True


class TestExtractCitationsFromText:
    """Test extract_citations_from_text function."""

    def test_extract_double_quotes(self) -> None:
        """Double quoted text is extracted."""
        text = 'He said "This is important information for the record"'
        result = extract_citations_from_text(text)
        assert any("important information" in c for c in result)

    def test_extract_single_quotes(self) -> None:
        """Single quoted text is extracted."""
        text = "He said 'This is key data for analysis'"
        result = extract_citations_from_text(text)
        assert any("key data" in c for c in result)

    def test_extract_french_quotes(self) -> None:
        """French quoted text is extracted."""
        text = "The report stated « Vietnam GDP grew 8% in 2024 » according to sources"
        result = extract_citations_from_text(text)
        assert any("Vietnam GDP grew 8%" in c for c in result)

    def test_extract_too_short_ignored(self) -> None:
        """Quotes shorter than 20 chars are ignored."""
        text = 'He said "ok" and left'
        result = extract_citations_from_text(text)
        assert not any(len(c) < 20 for c in result)

    def test_extract_no_citations(self) -> None:
        """Text without quotes returns empty list."""
        result = extract_citations_from_text("No quotes here")
        assert result == []


class TestVerifyCitationEvidencePairs:
    """Test verify_citation_evidence_pairs function."""

    def test_verify_empty_citations(self) -> None:
        """Empty citations returns empty results."""
        evidence = [{"content": "Test"}]
        result = verify_citation_evidence_pairs([], evidence)
        assert result == []

    def test_verify_valid_pair(self) -> None:
        """Valid citation-evidence pair is verified."""
        citations = [{"quote": "Vietnam GDP grew 8%", "source": "vnexpress"}]
        evidence = [{"content": "According to report, Vietnam GDP grew 8% in 2023"}]
        result = verify_citation_evidence_pairs(citations, evidence)
        assert len(result) == 1
        assert result[0]["is_valid"] is True

    def test_verify_invalid_pair(self) -> None:
        """Invalid citation-evidence pair is marked."""
        citations = [{"quote": "Completely different claim", "source": "unknown"}]
        evidence = [{"content": "Some other information"}]
        result = verify_citation_evidence_pairs(citations, evidence)
        assert len(result) == 1
        assert result[0]["is_valid"] is False


class TestFindMatchingEvidence:
    """Test _find_matching_evidence function."""

    def test_find_in_first_evidence(self) -> None:
        """Quote found in first evidence."""
        evidence = [
            {"content": "Vietnam GDP grew 8%"},
            {"content": "Other information"},
        ]
        result = _find_matching_evidence("Vietnam GDP grew", evidence)
        assert 0 in result

    def test_find_in_multiple(self) -> None:
        """Quote found in multiple evidence."""
        evidence = [
            {"content": "Some text with keyword"},
            {"content": "More text with keyword"},
            {"content": "Also has keyword here"},
        ]
        result = _find_matching_evidence("keyword", evidence)
        assert len(result) >= 1

    def test_find_not_found(self) -> None:
        """Quote not found returns empty list."""
        evidence = [
            {"content": "Some content"},
            {"content": "More content"},
        ]
        result = _find_matching_evidence("nonexistent quote here", evidence)
        assert result == []

    def test_find_normalizes_text(self) -> None:
        """Matching considers normalized text."""
        evidence = [{"content": "Vietnam GDP GREW 8%"}]
        result = _find_matching_evidence("vietnam gdp grew", evidence)
        assert 0 in result
