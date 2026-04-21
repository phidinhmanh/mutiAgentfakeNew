"""Tests for claim_extractor agent."""
from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from fake_news_detector.agents.claim_extractor import (
    _classify_claim,
    _parse_llm_response,
    extract_claims,
    filter_verifiable_claims,
    verify_facts_with_llm,
)


class TestExtractClaims:
    """Test extract_claims function."""

    def test_extract_claims_empty_string(self, mock_underthesea: Mock) -> None:
        """Empty string returns empty list."""
        result = extract_claims("")
        assert result == []

    def test_extract_claims_whitespace_only(self, mock_underthesea: Mock) -> None:
        """Whitespace-only returns empty list."""
        result = extract_claims("   \n\t   ")
        assert result == []

    def test_extract_claims_none_input(self, mock_underthesea: Mock) -> None:
        """None input returns empty list."""
        result = extract_claims(None)  # type: ignore
        assert result == []

    def test_extract_claims_short_sentences_ignored(
        self, mock_underthesea: Mock
    ) -> None:
        """Sentences shorter than 20 characters are filtered out."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "fake_news_detector.data.preprocessing.sent_tokenize",
                Mock(return_value=["Short.", "This is a longer sentence that matters."]),
            )
            result = extract_claims("Short. This is a longer sentence that matters.")
            assert len(result) == 1
            assert "longer sentence" in result[0]["text"]
            assert "Short" not in result[0]["text"]

    def test_extract_claims_normal_article(
        self, mock_underthesea: Mock, sample_vietnamese_text: str
    ) -> None:
        """Normal article extracts multiple claims."""
        result = extract_claims(sample_vietnamese_text)
        assert len(result) >= 1
        assert all("text" in c for c in result)
        assert all("type" in c for c in result)
        assert all("verifiable" in c for c in result)

    def test_extract_claims_fact_classification(
        self, mock_underthesea: Mock
    ) -> None:
        """FACT claims have verifiable=True."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "fake_news_detector.data.preprocessing.sent_tokenize",
                Mock(return_value=[
                    "Theo báo cáo của Bộ Y tế, số ca mắc COVID-19 giảm 30%.",
                ]),
            )
            result = extract_claims("Theo báo cáo của Bộ Y tế, số ca mắc COVID-19 giảm 30%.")
            assert len(result) == 1
            assert result[0]["type"] == "FACT"
            assert result[0]["verifiable"] is True

    def test_extract_claims_opinion_classification(
        self, mock_underthesea: Mock
    ) -> None:
        """OPINION claims have verifiable=False."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "fake_news_detector.data.preprocessing.sent_tokenize",
                Mock(return_value=["Tôi nghĩ đây là tin tốt cho Việt Nam."]),
            )
            result = extract_claims("Tôi nghĩ đây là tin tốt cho Việt Nam.")
            assert len(result) == 1
            assert result[0]["type"] == "OPINION"
            assert result[0]["verifiable"] is False


class TestClassifyClaim:
    """Test _classify_claim internal function."""

    def test_classify_fact_with_numbers(self) -> None:
        """Numbers in sentence indicate FACT."""
        result = _classify_claim("GDP tăng trưởng 8% trong năm 2023")
        assert result == "FACT"

    def test_classify_fact_with_report_indicators(self) -> None:
        """Report/analysis indicators indicate FACT."""
        result = _classify_claim("Theo báo cáo mới nhất của Bộ Y tế")
        assert result == "FACT"

    def test_classify_fact_with_data_indicators(self) -> None:
        """Data/statistics indicators indicate FACT."""
        result = _classify_claim("Số liệu cho thấy tăng trưởng kinh tế")
        assert result == "FACT"

    def test_classify_opinion_with_personal_indicators(self) -> None:
        """Personal opinion indicators indicate OPINION."""
        result = _classify_claim("Tôi nghĩ đây là tin tốt")
        assert result == "OPINION"

    def test_classify_opinion_with_uncertainty_indicators(self) -> None:
        """Uncertainty indicators indicate OPINION."""
        result = _classify_claim("Có lẽ chúng ta nên thận trọng")
        assert result == "OPINION"

    def test_classify_opinion_with_judgment_indicators(self) -> None:
        """Judgment words indicate OPINION."""
        result = _classify_claim("Đây là tin rất tốt cho Việt Nam")
        assert result == "OPINION"

    def test_classify_fact_wins_over_opinion(self) -> None:
        """FACT indicators outweigh OPINION indicators."""
        result = _classify_claim(
            "Theo báo cáo, GDP tăng trưởng 8% nhưng tôi nghĩ còn tốt hơn nữa"
        )
        assert result == "FACT"

    def test_classify_opinion_wins_over_fact(self) -> None:
        """OPINION indicators outweigh FACT indicators."""
        result = _classify_claim(
            "Tôi tin rằng theo số liệu này chúng ta nên suy nghĩ kỹ"
        )
        assert result == "OPINION"

    def test_classify_default_with_digits(self) -> None:
        """Digits break tie in favor of FACT."""
        result = _classify_claim("Khẳng định nào đó có số 123")
        assert result == "FACT"

    def test_classify_default_without_digits(self) -> None:
        """No digits leads to OPINION on tie."""
        result = _classify_claim("Một khẳng định đơn giản")
        assert result == "OPINION"

    def test_classify_empty_string(self) -> None:
        """Empty string returns OPINION (no indicators)."""
        result = _classify_claim("")
        assert result == "OPINION"


class TestVerifyFactsWithLLM:
    """Test verify_facts_with_llm function."""

    def test_verify_facts_empty_claims(self, mock_nvidia_client: Mock) -> None:
        """Empty claims list returns empty list."""
        result = verify_facts_with_llm([], mock_nvidia_client)
        assert result == []

    def test_verify_facts_single_fact_claim(
        self, mock_nvidia_client: Mock
    ) -> None:
        """Single FACT claim gets verified."""
        claims = [{"text": "GDP tăng 8%", "type": "FACT", "verifiable": True}]
        result = verify_facts_with_llm(claims, mock_nvidia_client)
        assert len(result) == 1
        assert "verified" in result[0]
        assert result[0]["verified"] is not None

    def test_verify_facts_opinion_skipped(self, mock_nvidia_client: Mock) -> None:
        """OPINION claims are skipped (verified=None)."""
        claims = [
            {"text": "Tôi nghĩ đây là tin tốt", "type": "OPINION", "verifiable": False}
        ]
        result = verify_facts_with_llm(claims, mock_nvidia_client)
        assert len(result) == 1
        assert result[0]["verified"] is None

    def test_verify_facts_mixed_claims(
        self, mock_nvidia_client: Mock
    ) -> None:
        """Mixed FACT/OPINION claims are handled correctly."""
        claims = [
            {"text": "GDP tăng 8%", "type": "FACT", "verifiable": True},
            {"text": "Tôi nghĩ đây là tin tốt", "type": "OPINION", "verifiable": False},
        ]
        result = verify_facts_with_llm(claims, mock_nvidia_client)
        assert len(result) == 2
        assert result[0]["verified"] is not None
        assert result[1]["verified"] is None

    def test_verify_facts_llm_error_returns_none(
        self, mock_nvidia_client: Mock
    ) -> None:
        """LLM error returns verified=None for that claim."""
        mock_nvidia_client.invoke.side_effect = Exception("API Error")
        claims = [{"text": "GDP tăng 8%", "type": "FACT", "verifiable": True}]
        result = verify_facts_with_llm(claims, mock_nvidia_client)
        assert len(result) == 1
        assert result[0]["verified"] is None


class TestParseLLMResponse:
    """Test _parse_llm_response function."""

    def test_parse_valid_json(self) -> None:
        """Valid JSON returns parsed dict."""
        response = '{"verdict": "REAL", "confidence": 0.8}'
        result = _parse_llm_response(response)
        assert result["verdict"] == "REAL"
        assert result["confidence"] == 0.8

    def test_parse_json_with_extra_text(self) -> None:
        """JSON with surrounding text is extracted correctly."""
        response = 'Here is the response: {"verdict": "FAKE", "confidence": 0.9}'
        result = _parse_llm_response(response)
        assert result["verdict"] == "FAKE"

    def test_parse_invalid_json(self) -> None:
        """Invalid JSON returns empty dict."""
        result = _parse_llm_response("This is not JSON")
        assert result == {}

    def test_parse_empty_string(self) -> None:
        """Empty string returns empty dict."""
        result = _parse_llm_response("")
        assert result == {}


class TestFilterVerifiableClaims:
    """Test filter_verifiable_claims function."""

    def test_filter_empty_list(self) -> None:
        """Empty list returns empty list."""
        result = filter_verifiable_claims([])
        assert result == []

    def test_filter_only_facts(self) -> None:
        """Only FACT claims with verifiable=True are returned."""
        claims = [
            {"text": "GDP tăng 8%", "type": "FACT", "verifiable": True},
            {"text": "GDP giảm", "type": "FACT", "verifiable": True},
        ]
        result = filter_verifiable_claims(claims)
        assert len(result) == 2

    def test_filter_removes_opinions(self) -> None:
        """OPINION claims are filtered out."""
        claims = [
            {"text": "GDP tăng 8%", "type": "FACT", "verifiable": True},
            {"text": "Tôi nghĩ đây là tin tốt", "type": "OPINION", "verifiable": False},
        ]
        result = filter_verifiable_claims(claims)
        assert len(result) == 1
        assert result[0]["type"] == "FACT"

    def test_filter_removes_non_verifiable_facts(self) -> None:
        """FACT claims with verifiable=False are filtered out."""
        claims = [
            {"text": "GDP tăng 8%", "type": "FACT", "verifiable": True},
            {"text": "GDP giảm", "type": "FACT", "verifiable": False},
        ]
        result = filter_verifiable_claims(claims)
        assert len(result) == 1

    def test_filter_handles_missing_keys(self) -> None:
        """Claims missing type/verifiable keys are handled."""
        claims = [{"text": "Some text"}]
        result = filter_verifiable_claims(claims)
        assert len(result) == 0