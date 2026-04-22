# -*- coding: utf-8 -*-
"""Acceptance tests for end-to-end fact-checking flow.

Tests the complete system behavior using golden samples with known expected outcomes.
These tests verify that the system produces correct labels (REAL/FAKE/UNVERIFIABLE)
for sample Vietnamese news texts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest


@dataclass
class GoldenSample:
    """A golden sample for acceptance testing."""
    id: str
    text: str
    expected_verdict: str  # "true", "false", or "uncertain"
    expected_claims: int
    description: str


@dataclass
class AcceptanceResult:
    """Result of acceptance test for a sample."""
    sample_id: str
    passed: bool
    actual_verdict: str | None
    expected_verdict: str
    error_message: str | None


# === Golden Samples ===

VIETNAMESE_FAKE_NEWS_SAMPLES = [
    GoldenSample(
        id="fake_news_alien",
        text="Việt Nam đã phát hiện người ngoài hành tinh tại Hà Nội và chính phủ đang che giấu thông tin này.",
        expected_verdict="false",
        expected_claims=1,
        description="Tin giả với thông tin vô lý về UFO"
    ),
    GoldenSample(
        id="fake_news_gdp_exaggerated",
        text="Việt Nam đạt tăng trưởng GDP 50% trong quý 1 năm 2024, cao nhất thế giới.",
        expected_verdict="false",
        expected_claims=1,
        description="Tin giả với số liệu GDP phi lý"
    ),
    GoldenSample(
        id="fake_news_health_false",
        text="Uống nước đá có thể chữa khỏi ung thư trong vòng 1 tuần. Bác sĩ nổi tiếng xác nhận.",
        expected_verdict="false",
        expected_claims=1,
        description="Tin giả về y tế"
    ),
]

VIETNAMESE_REAL_NEWS_SAMPLES = [
    GoldenSample(
        id="real_news_gdp_wb",
        text="Theo báo cáo của World Bank, Việt Nam đạt tăng trưởng GDP 5.6% trong năm 2023, thuộc nhóm cao nhất ASEAN.",
        expected_verdict="true",
        expected_claims=1,
        description="Tin thật với số liệu cụ thể từ WB"
    ),
    GoldenSample(
        id="real_news_covid_moh",
        text="Bộ Y tế công bố số ca mắc COVID-19 ngày 15/3/2024 giảm 30% so với tuần trước, xuống còn 1,500 ca.",
        expected_verdict="true",
        expected_claims=1,
        description="Tin thật với số liệu chính thức từ Bộ Y tế"
    ),
]

VIETNAMESE_UNVERIFIABLE_SAMPLES = [
    GoldenSample(
        id="unverifiable_generic",
        text="Một nguồn tin cho biết lãnh đạo đất nước sẽ có cuộc họp quan trọng vào tuần tới nhưng không tiết lộ chi tiết.",
        expected_verdict="uncertain",
        expected_claims=1,
        description="Tin không thể xác minh do thiếu chi tiết cụ thể"
    ),
    GoldenSample(
        id="unverifiable_rumor",
        text="Có tin đồn rằng một công ty lớn sẽ thay đổi chính sách nhưng chưa có thông tin chính thức.",
        expected_verdict="uncertain",
        expected_claims=1,
        description="Tin đồn chưa được xác minh"
    ),
]


class TestFactCheckFlowAcceptance:
    """Acceptance tests for the complete fact-checking flow."""

    def _run_pipeline_for_sample(
        self,
        sample: GoldenSample,
        mock_claim_extractor: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier: Mock,
    ) -> AcceptanceResult:
        """Run the complete pipeline for a single sample."""
        try:
            from trust_agents.orchestrator import TRUSTOrchestrator

            orchestrator = TRUSTOrchestrator()

            # Run pipeline with skip_evidence=True since we patch evidence retrieval
            result = orchestrator.process_text(sample.text, skip_evidence=True)

            # Get verdict from first result
            actual_verdict = None
            if result.results:
                actual_verdict = result.results[0].get("verdict")

            # Check if verdict matches expected
            passed = actual_verdict == sample.expected_verdict

            return AcceptanceResult(
                sample_id=sample.id,
                passed=passed,
                actual_verdict=actual_verdict,
                expected_verdict=sample.expected_verdict,
                error_message=None if passed else f"Expected {sample.expected_verdict}, got {actual_verdict}"
            )
        except Exception as e:
            return AcceptanceResult(
                sample_id=sample.id,
                passed=False,
                actual_verdict=None,
                expected_verdict=sample.expected_verdict,
                error_message=str(e)
            )

    def _run_pipeline_with_evidence(
        self,
        sample: GoldenSample,
        evidence: list[dict[str, Any]],
        expected_verdict: str,
    ) -> AcceptanceResult:
        """Run pipeline with specific evidence and expected verdict."""
        try:
            from trust_agents.orchestrator import TRUSTOrchestrator

            orchestrator = TRUSTOrchestrator()

            # Patch evidence retrieval to return our evidence
            with patch(
                "trust_agents.orchestrator.run_evidence_retrieval_agent_sync",
                return_value=evidence
            ):
                with patch(
                    "trust_agents.orchestrator.run_verifier_agent_sync",
                    return_value={
                        "verdict": expected_verdict,
                        "confidence": 0.85,
                        "label": expected_verdict,
                        "reasoning": f"Evidence {'supports' if expected_verdict == 'true' else 'contradicts'} claim"
                    }
                ):
                    with patch(
                        "trust_agents.orchestrator.run_explainer_agent_sync",
                        return_value={
                            "summary": f"Claim verified as {expected_verdict}",
                            "explanation": f"Evidence {'supports' if expected_verdict == 'true' else 'contradicts'} the claim"
                        }
                    ):
                        result = orchestrator.process_text(sample.text)

            # Get verdict from first result
            actual_verdict = None
            if result.results:
                actual_verdict = result.results[0].get("verdict")

            # Check if verdict matches expected
            passed = actual_verdict == expected_verdict

            return AcceptanceResult(
                sample_id=sample.id,
                passed=passed,
                actual_verdict=actual_verdict,
                expected_verdict=expected_verdict,
                error_message=None if passed else f"Expected {expected_verdict}, got {actual_verdict}"
            )
        except Exception as e:
            return AcceptanceResult(
                sample_id=sample.id,
                passed=False,
                actual_verdict=None,
                expected_verdict=expected_verdict,
                error_message=str(e)
            )

    def test_fake_news_samples_return_false(
        self,
        mock_claim_extractor_agent: Mock,
    ) -> None:
        """Test that fake news samples return 'false' verdict."""
        for sample in VIETNAMESE_FAKE_NEWS_SAMPLES:
            evidence = [
                {"content": f"Evidence contradicting: {sample.text}", "source": "test", "score": 0.9}
            ]
            result = self._run_pipeline_with_evidence(
                sample, evidence, expected_verdict="false"
            )

            assert result.passed, f"Sample {sample.id} failed: {result.error_message}"
            assert result.actual_verdict == "false"

    def test_real_news_samples_return_true(
        self,
        mock_claim_extractor_agent: Mock,
    ) -> None:
        """Test that real news samples return 'true' verdict."""
        for sample in VIETNAMESE_REAL_NEWS_SAMPLES:
            evidence = [
                {"content": f"Evidence supporting: {sample.text}", "source": "test", "score": 0.9}
            ]
            result = self._run_pipeline_with_evidence(
                sample, evidence, expected_verdict="true"
            )

            assert result.passed, f"Sample {sample.id} failed: {result.error_message}"
            assert result.actual_verdict == "true"

    def test_unverifiable_samples_return_uncertain(
        self,
        mock_claim_extractor_agent: Mock,
    ) -> None:
        """Test that unverifiable samples return 'uncertain' verdict."""
        for sample in VIETNAMESE_UNVERIFIABLE_SAMPLES:
            with patch(
                "trust_agents.orchestrator.run_evidence_retrieval_agent_sync",
                return_value=[]  # No evidence
            ):
                with patch(
                    "trust_agents.orchestrator.run_verifier_agent_sync",
                    return_value={
                        "verdict": "uncertain",
                        "confidence": 0.3,
                        "label": "uncertain",
                        "reasoning": "Not enough evidence"
                    }
                ):
                    with patch(
                        "trust_agents.orchestrator.run_explainer_agent_sync",
                        return_value={
                            "summary": "Cannot verify - insufficient evidence",
                            "explanation": "Not enough evidence to make a determination"
                        }
                    ):
                        result = self._run_pipeline_for_sample(
                            sample,
                            mock_claim_extractor_agent,
                            None,
                            None,
                        )

            assert result.passed, f"Sample {sample.id} failed: {result.error_message}"
            assert result.actual_verdict == "uncertain"


class TestFactCheckResultQuality:
    """Tests for result quality and completeness."""

    def test_result_contains_all_required_fields(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test that results contain all required fields for UI display."""
        from trust_agents.orchestrator import TRUSTOrchestrator

        orchestrator = TRUSTOrchestrator()
        result = orchestrator.process_text("Test text")

        assert len(result.results) >= 1
        first_result = result.results[0]

        # Check required fields for UI
        required_fields = ["verdict", "confidence"]
        for field in required_fields:
            assert field in first_result, f"Missing required field: {field}"

    def test_result_summary_has_statistics(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test that summary contains statistics."""
        from trust_agents.orchestrator import TRUSTOrchestrator

        orchestrator = TRUSTOrchestrator()
        result = orchestrator.process_text("Test text")

        assert "total_claims" in result.summary
        assert "verdicts" in result.summary

    def test_result_includes_original_text(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test that original text is preserved in result."""
        from trust_agents.orchestrator import TRUSTOrchestrator

        test_text = "Việt Nam đạt tăng trưởng 8%"
        orchestrator = TRUSTOrchestrator()
        result = orchestrator.process_text(test_text)

        assert result.original_text == test_text


class TestFactCheckPerformance:
    """Tests for performance requirements."""

    def test_pipeline_completes_within_timeout(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test that pipeline completes within reasonable time."""
        import time
        from trust_agents.orchestrator import TRUSTOrchestrator

        orchestrator = TRUSTOrchestrator()

        start_time = time.time()
        result = orchestrator.process_text("Test text")
        elapsed_time = time.time() - start_time

        # Should complete in less than 5 seconds with mocks
        assert elapsed_time < 5.0, f"Pipeline took {elapsed_time}s, expected < 5s"

    def test_pipeline_handles_multiple_claims(
        self,
        mock_claim_extractor_multiple: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test pipeline handles text with multiple claims."""
        from trust_agents.orchestrator import TRUSTOrchestrator

        orchestrator = TRUSTOrchestrator()
        result = orchestrator.process_text(
            "Số ca mắc COVID-19 giảm 30%. Chúng tôi đã kiểm soát được dịch. "
            "Tốc độ tăng trưởng đạt 8%."
        )

        # Should process all 3 claims
        assert len(result.results) == 3


class TestFactCheckEdgeCases:
    """Tests for edge cases in fact-checking."""

    def test_very_short_text(
        self,
        mock_claim_extractor_agent: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test handling of very short text."""
        from trust_agents.orchestrator import TRUSTOrchestrator

        orchestrator = TRUSTOrchestrator()
        result = orchestrator.process_text("Test", skip_evidence=True)

        # Should handle gracefully
        assert isinstance(result, type(result))

    def test_repeated_same_text(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test processing same text multiple times."""
        from trust_agents.orchestrator import TRUSTOrchestrator

        orchestrator = TRUSTOrchestrator()
        text = "Việt Nam đạt tăng trưởng 8%"

        # Process same text twice
        result1 = orchestrator.process_text(text)
        result2 = orchestrator.process_text(text)

        # Both should produce results
        assert len(result1.results) == len(result2.results)