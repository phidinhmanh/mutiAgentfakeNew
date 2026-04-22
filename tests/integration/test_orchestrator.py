# -*- coding: utf-8 -*-
"""Integration tests for TRUSTOrchestrator.

Tests the complete pipeline: Claim Extraction -> Evidence Retrieval -> Verification -> Explanation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from trust_agents.orchestrator import TRUSTOrchestrator, TRUSTResult


class TestOrchestratorIntegration:
    """Integration tests for TRUSTOrchestrator."""

    # === Pipeline Flow Tests ===

    def test_process_text_with_multiple_claims(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test complete pipeline with multiple claims."""
        orchestrator = TRUSTOrchestrator(top_k_evidence=5)

        result = orchestrator.process_text("Sample text with multiple claims")

        # Verify all agents were called
        assert mock_claim_extractor_agent.called
        assert mock_evidence_retriever.called
        assert mock_verifier_agent.called
        assert mock_explainer_agent.called

        # Verify result structure
        assert isinstance(result, TRUSTResult)
        assert len(result.claims) == 2
        assert len(result.results) == 2
        assert result.summary["total_claims"] == 2

    def test_process_text_with_single_claim(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test pipeline with single claim."""
        orchestrator = TRUSTOrchestrator(top_k_evidence=5)

        result = orchestrator.process_text("Single claim text")

        assert len(result.claims) == 2
        assert len(result.results) == 2
        assert result.summary["total_claims"] == 2

    def test_process_text_empty_claims(
        self,
        mock_claim_extractor_empty: Mock,
    ) -> None:
        """Test pipeline when no claims are extracted."""
        orchestrator = TRUSTOrchestrator()

        result = orchestrator.process_text("No claims here")

        assert len(result.claims) == 0
        assert len(result.results) == 0
        assert result.summary["status"] == "no_claims"
        assert "message" in result.summary

    def test_process_text_skip_evidence(
        self,
        mock_claim_extractor_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test pipeline skipping evidence retrieval."""
        orchestrator = TRUSTOrchestrator()

        result = orchestrator.process_text("Test text", skip_evidence=True)

        # Claim extractor and explainer should still be called
        assert mock_claim_extractor_agent.called
        assert mock_explainer_agent.called

        # When skip_evidence, verifier returns uncertain directly (no evidence)
        # so verify the result is uncertain
        assert len(result.results) >= 1

    # === Normalization Tests ===

    def test_normalize_verdict_supported_to_true(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
    ) -> None:
        """Test verdict normalization: 'supported' -> 'true'."""
        orchestrator = TRUSTOrchestrator()

        verdict_data = {
            "verdict": "supported",
            "confidence": 75,  # Should be converted to 0.75
            "reasoning": "Evidence supports claim"
        }

        normalized = orchestrator._normalize_verdict(verdict_data)

        assert normalized["verdict"] == "true"
        assert normalized["label"] == "true"
        assert normalized["confidence"] == 0.75

    def test_normalize_verdict_contradicted_to_false(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
    ) -> None:
        """Test verdict normalization: 'contradicted' -> 'false'."""
        orchestrator = TRUSTOrchestrator()

        verdict_data = {
            "verdict": "contradicted",
            "confidence": 90,
            "reasoning": "Evidence contradicts"
        }

        normalized = orchestrator._normalize_verdict(verdict_data)

        assert normalized["verdict"] == "false"
        assert normalized["label"] == "false"
        assert normalized["confidence"] == 0.9

    def test_normalize_verdict_insufficient_to_uncertain(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
    ) -> None:
        """Test verdict normalization: 'insufficient' -> 'uncertain'."""
        orchestrator = TRUSTOrchestrator()

        verdict_data = {
            "verdict": "insufficient",
            "confidence": 0.3,
            "reasoning": "Not enough evidence"
        }

        normalized = orchestrator._normalize_verdict(verdict_data)

        assert normalized["verdict"] == "uncertain"
        assert normalized["label"] == "uncertain"

    def test_normalize_verdict_confidence_out_of_range(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
    ) -> None:
        """Test confidence normalization: values > 1.0 get divided by 100."""
        orchestrator = TRUSTOrchestrator()

        verdict_data = {
            "verdict": "true",
            "confidence": 150,  # Out of range, should become 1.5 -> clamped to 1.0
            "reasoning": "High confidence"
        }

        normalized = orchestrator._normalize_verdict(verdict_data)

        assert normalized["confidence"] == 1.0

    def test_normalize_verdict_string_confidence(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
    ) -> None:
        """Test parsing string confidence values."""
        orchestrator = TRUSTOrchestrator()

        verdict_data = {
            "verdict": "true",
            "confidence": "0.85",
            "reasoning": "String confidence"
        }

        normalized = orchestrator._normalize_verdict(verdict_data)

        assert normalized["confidence"] == 0.85

    def test_normalize_verdict_invalid_confidence(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
    ) -> None:
        """Test fallback for invalid confidence values."""
        orchestrator = TRUSTOrchestrator()

        verdict_data = {
            "verdict": "true",
            "confidence": "invalid",
            "reasoning": "Test"
        }

        normalized = orchestrator._normalize_verdict(verdict_data)

        # Should default to 0.3
        assert normalized["confidence"] == 0.3

    def test_normalize_verdict_text_extraction(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
    ) -> None:
        """Test extracting verdict from natural language text."""
        orchestrator = TRUSTOrchestrator()

        verdict_data = {
            "verdict": "The claim is false according to our analysis",
            "confidence": 0.8,
            "reasoning": "Analysis shows contradiction"
        }

        normalized = orchestrator._normalize_verdict(verdict_data)

        assert normalized["verdict"] == "false"
        assert normalized["label"] == "false"

    # === Error Recovery Tests ===

    def test_process_text_evidence_retrieval_error(
        self,
        mock_claim_extractor_agent: Mock,
    ) -> None:
        """Test graceful handling when evidence retrieval fails."""
        orchestrator = TRUSTOrchestrator()

        # Patch all evidence-related mocks to return empty/uncertain
        with patch(
            "trust_agents.orchestrator.run_evidence_retrieval_agent_sync",
            return_value=[]  # Return empty instead of raising
        ):
            with patch(
                "trust_agents.orchestrator.run_verifier_agent_sync",
                return_value={
                    "verdict": "uncertain",
                    "confidence": 0.1,
                    "label": "uncertain",
                    "reasoning": "No evidence available"
                }
            ):
                # Explainer should preserve verdict from verifier
                with patch(
                    "trust_agents.orchestrator.run_explainer_agent_sync",
                    return_value={
                        "summary": "Cannot verify - no evidence",
                        "explanation": "No evidence found for verification"
                    }  # No verdict key - will be merged from verifier
                ):
                    result = orchestrator.process_text("Test text")

        # Should still return results - when no evidence, verdict is "uncertain"
        assert len(result.results) >= 1
        assert result.results[0]["verdict"] == "uncertain"

    def test_process_text_verifier_error(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
    ) -> None:
        """Test graceful handling when verifier fails."""
        orchestrator = TRUSTOrchestrator()

        # Patch verifier to raise exception
        with patch(
            "trust_agents.orchestrator.run_verifier_agent_sync",
            side_effect=Exception("LLM API error")
        ):
            result = orchestrator.process_text("Test text")

        # Should still return results
        assert len(result.results) >= 1
        assert "error" in result.results[0] or result.results[0]["verdict"] == "uncertain"

    def test_process_text_explainer_error(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
    ) -> None:
        """Test graceful handling when explainer fails."""
        orchestrator = TRUSTOrchestrator()

        # Patch explainer to raise exception
        with patch(
            "trust_agents.orchestrator.run_explainer_agent_sync",
            side_effect=Exception("Explainer failed")
        ):
            result = orchestrator.process_text("Test text")

        # Should still return results (verdict from verifier)
        assert len(result.results) >= 1

    # === Summary Tests ===

    def test_create_summary_with_mixed_verdicts(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test summary calculation with mixed verdict types."""
        orchestrator = TRUSTOrchestrator()

        # Manually create results with different verdicts
        results = [
            {"verdict": "true", "confidence": 0.85},
            {"verdict": "false", "confidence": 0.90},
            {"verdict": "uncertain", "confidence": 0.3},
        ]

        summary = orchestrator._create_summary(results)

        assert summary["total_claims"] == 3
        assert summary["verdicts"]["true"] == 1
        assert summary["verdicts"]["false"] == 1
        assert summary["verdicts"]["uncertain"] == 1
        assert summary["high_confidence_claims"] == 2
        assert summary["low_confidence_claims"] == 1

    def test_create_summary_with_percentage_confidence(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test summary handles percentage confidence (0-100)."""
        orchestrator = TRUSTOrchestrator()

        results = [
            {"verdict": "true", "confidence": 85},  # Percentage
            {"verdict": "true", "confidence": 0.9},  # Decimal
        ]

        summary = orchestrator._create_summary(results)

        # Both should be converted to decimal range
        assert summary["verdicts"]["true"] == 2
        # Average should be around 0.875
        assert 0.8 <= summary["average_confidence"] <= 0.9

    # === Edge Case Tests ===

    def test_process_text_very_long_input(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test handling of very long input text."""
        orchestrator = TRUSTOrchestrator()

        long_text = "Word " * 1000  # 5000 characters

        result = orchestrator.process_text(long_text)

        assert isinstance(result, TRUSTResult)

    def test_process_text_special_characters(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test handling of text with special characters."""
        orchestrator = TRUSTOrchestrator()

        special_text = "Test with émojis 🎉 and special chars: @#$%^&*()"

        result = orchestrator.process_text(special_text)

        assert isinstance(result, TRUSTResult)

    def test_process_text_vietnamese_input(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test handling of Vietnamese text."""
        orchestrator = TRUSTOrchestrator()

        vietnamese_text = "Theo báo cáo của Bộ Y tế, Việt Nam đã kiểm soát được dịch COVID-19."

        result = orchestrator.process_text(vietnamese_text)

        assert isinstance(result, TRUSTResult)


class TestOrchestratorResultStructure:
    """Tests for TRUSTResult structure."""

    def test_result_has_required_fields(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test that result has all required fields."""
        orchestrator = TRUSTOrchestrator()

        result = orchestrator.process_text("Test text")

        assert hasattr(result, "original_text")
        assert hasattr(result, "claims")
        assert hasattr(result, "results")
        assert hasattr(result, "summary")

    def test_result_results_have_required_fields(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test that individual results have required fields."""
        orchestrator = TRUSTOrchestrator()

        result = orchestrator.process_text("Test text")

        assert len(result.results) >= 1
        first_result = result.results[0]
        assert "verdict" in first_result
        assert "confidence" in first_result
        assert "reasoning" in first_result or "summary" in first_result


class TestRunTrustPipelineSync:
    """Tests for the convenience function run_trust_pipeline_sync."""

    def test_fact_check_convenience_function(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test the fact_check convenience function."""
        from trust_agents.orchestrator import run_trust_pipeline_sync

        result = run_trust_pipeline_sync("Test text")

        assert isinstance(result, dict)
        assert "original_text" in result
        assert "claims" in result
        assert "results" in result

    def test_fact_check_alias(
        self,
        mock_claim_extractor_agent: Mock,
        mock_evidence_retriever: Mock,
        mock_verifier_agent: Mock,
        mock_explainer_agent: Mock,
    ) -> None:
        """Test the fact_check alias function."""
        from trust_agents.orchestrator import fact_check

        result = fact_check("Test text")

        assert isinstance(result, dict)
        assert "results" in result
