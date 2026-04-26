"""Integration tests for ResearchTRUSTOrchestrator.

Tests the research pipeline: Decomposition -> Delphi Jury -> Logic Aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock, patch

import pytest


@dataclass
class MockDecomposedClaim:
    """Mock for DecomposedClaim dataclass."""
    original_claim: str
    atomic_claims: list[str]
    logic_structure: dict[str, Any]
    complexity_score: float
    causal_edges: list[tuple[str, str]]


class TestResearchOrchestratorIntegration:
    """Integration tests for ResearchTRUSTOrchestrator."""

    @pytest.fixture
    def mock_decomposer(self, monkeypatch: pytest.MonkeyPatch) -> Mock:
        """Mock DecomposerAgent."""
        mock_decomposed = MockDecomposedClaim(
            original_claim="Test claim",
            atomic_claims=[
                "Việt Nam đạt tăng trưởng 8%",
                "Tăng trưởng này cao nhất ASEAN"
            ],
            logic_structure={"type": "conjunction"},
            complexity_score=0.5,
            causal_edges=[]
        )

        mock = Mock()
        mock.decompose = Mock(return_value=mock_decomposed)

        monkeypatch.setattr(
            "trust_agents.orchestrator_research.DecomposerAgent",
            lambda: mock
        )
        return mock

    @pytest.fixture
    def mock_logic_aggregator(self, monkeypatch: pytest.MonkeyPatch) -> Mock:
        """Mock LogicAggregator."""
        mock = Mock()
        mock.aggregate = Mock(return_value={
            "verdict": "true",
            "confidence": 0.85,
            "reasoning": "All atomic claims verified"
        })

        monkeypatch.setattr(
            "trust_agents.orchestrator_research.LogicAggregator",
            lambda: mock
        )
        return mock

    @pytest.fixture
    def mock_delphi_jury(self, monkeypatch: pytest.MonkeyPatch) -> Mock:
        """Mock DelphiJury."""
        mock = Mock()
        mock.verify_with_jury = Mock(return_value={
            "verdict": "true",
            "confidence": 0.85,
            "reasoning": "Jury consensus",
            "jury_verdicts": [
                {"agent": "agent_1", "verdict": "true", "confidence": 0.8},
                {"agent": "agent_2", "verdict": "true", "confidence": 0.9},
            ]
        })

        monkeypatch.setattr(
            "trust_agents.orchestrator_research.DelphiJury",
            lambda: mock
        )
        return mock

    def test_research_pipeline_decomposition(
        self,
        mock_decomposer: Mock,
        mock_logic_aggregator: Mock,
        mock_delphi_jury: Mock,
    ) -> None:
        """Test that research pipeline decomposes claims correctly."""
        # Import here to apply patches
        from trust_agents.orchestrator_research import ResearchTRUSTOrchestrator

        orchestrator = ResearchTRUSTOrchestrator(
            use_delphi_jury=True,
            top_k_evidence=5
        )

        # Patch evidence retrieval
        with patch(
            "trust_agents.orchestrator_research.run_evidence_retrieval_agent_sync",
            return_value=[
                {"content": "Test evidence", "source": "test", "score": 0.9}
            ]
        ):
            with patch(
                "trust_agents.orchestrator_research.run_explainer_agent_sync",
                return_value={"summary": "Test", "explanation": "Test"}
            ):
                result = orchestrator.process_text("Test text")

        assert mock_decomposer.decompose.called
        assert len(result.atomic_verdicts) >= 1

    def test_research_pipeline_delphi_jury_used(
        self,
        mock_decomposer: Mock,
        mock_logic_aggregator: Mock,
        mock_delphi_jury: Mock,
    ) -> None:
        """Test that Delphi jury is used when enabled."""
        from trust_agents.orchestrator_research import ResearchTRUSTOrchestrator

        orchestrator = ResearchTRUSTOrchestrator(
            use_delphi_jury=True,
            top_k_evidence=5
        )

        with patch(
            "trust_agents.orchestrator_research.run_evidence_retrieval_agent_sync",
            return_value=[
                {"content": "Test evidence", "source": "test", "score": 0.9}
            ]
        ):
            with patch(
                "trust_agents.orchestrator_research.run_explainer_agent_sync",
                return_value={"summary": "Test", "explanation": "Test"}
            ):
                orchestrator.process_text("Test text")

        # Verify jury was called for each atomic claim
        assert mock_delphi_jury.verify_with_jury.call_count >= 1

    def test_research_pipeline_without_delphi(
        self,
        mock_decomposer: Mock,
        mock_logic_aggregator: Mock,
    ) -> None:
        """Test pipeline without Delphi jury (fallback to single verifier)."""
        from trust_agents.orchestrator_research import ResearchTRUSTOrchestrator

        orchestrator = ResearchTRUSTOrchestrator(
            use_delphi_jury=False,
            top_k_evidence=5
        )

        with patch(
            "trust_agents.orchestrator_research.run_evidence_retrieval_agent_sync",
            return_value=[
                {"content": "Test evidence", "source": "test", "score": 0.9}
            ]
        ):
            with patch(
                "trust_agents.agents.verifier.run_verifier_agent_sync",
                return_value={
                    "verdict": "true",
                    "confidence": 0.85,
                    "reasoning": "Test"
                }
            ):
                with patch(
                    "trust_agents.orchestrator_research.run_explainer_agent_sync",
                    return_value={"summary": "Test", "explanation": "Test"}
                ):
                    result = orchestrator.process_text("Test text")

        assert len(result.atomic_verdicts) >= 1

    def test_research_pipeline_logic_aggregation(
        self,
        mock_decomposer: Mock,
        mock_logic_aggregator: Mock,
        mock_delphi_jury: Mock,
    ) -> None:
        """Test that logic aggregation is performed."""
        from trust_agents.orchestrator_research import ResearchTRUSTOrchestrator

        orchestrator = ResearchTRUSTOrchestrator(
            use_delphi_jury=True,
            top_k_evidence=5
        )

        with patch(
            "trust_agents.orchestrator_research.run_evidence_retrieval_agent_sync",
            return_value=[
                {"content": "Test evidence", "source": "test", "score": 0.9}
            ]
        ):
            with patch(
                "trust_agents.orchestrator_research.run_explainer_agent_sync",
                return_value={"summary": "Test", "explanation": "Test"}
            ):
                result = orchestrator.process_text("Test text")

        assert mock_logic_aggregator.aggregate.called
        assert "verdict" in result.logic_aggregation

    def test_research_pipeline_skip_evidence(
        self,
        mock_decomposer: Mock,
        mock_logic_aggregator: Mock,
        mock_delphi_jury: Mock,
    ) -> None:
        """Test skipping evidence retrieval."""
        from trust_agents.orchestrator_research import ResearchTRUSTOrchestrator

        orchestrator = ResearchTRUSTOrchestrator(
            use_delphi_jury=True,
            top_k_evidence=5
        )

        with patch(
            "trust_agents.orchestrator_research.run_explainer_agent_sync",
            return_value={"summary": "Test", "explanation": "Test"}
        ):
            result = orchestrator.process_text("Test text", skip_evidence=True)

        assert len(result.atomic_verdicts) >= 1

    def test_research_pipeline_metadata(
        self,
        mock_decomposer: Mock,
        mock_logic_aggregator: Mock,
        mock_delphi_jury: Mock,
    ) -> None:
        """Test that metadata is correctly collected."""
        from trust_agents.orchestrator_research import ResearchTRUSTOrchestrator

        orchestrator = ResearchTRUSTOrchestrator(
            use_delphi_jury=True,
            top_k_evidence=5
        )

        with patch(
            "trust_agents.orchestrator_research.run_evidence_retrieval_agent_sync",
            return_value=[
                {"content": "Test evidence", "source": "test", "score": 0.9}
            ]
        ):
            with patch(
                "trust_agents.orchestrator_research.run_explainer_agent_sync",
                return_value={"summary": "Test", "explanation": "Test"}
            ):
                result = orchestrator.process_text("Test text")

        assert hasattr(result, "metadata")
        assert "num_atomic_claims" in result.metadata
        assert "complexity_score" in result.metadata
        assert "used_delphi_jury" in result.metadata


class TestResearchOrchestratorResultStructure:
    """Tests for ResearchTRUSTResult structure."""

    def test_result_has_required_fields(self) -> None:
        """Test that result has all required fields."""
        from trust_agents.orchestrator_research import ResearchTRUSTResult

        # Verify dataclass has expected fields
        result = ResearchTRUSTResult(
            original_text="test",
            decomposed_claim=MockDecomposedClaim(
                original_claim="test",
                atomic_claims=[],
                logic_structure={},
                complexity_score=0.0,
                causal_edges=[]
            ),
            atomic_verdicts=[],
            logic_aggregation={"verdict": "uncertain", "confidence": 0.0},
            final_verdict={},
            metadata={}
        )

        assert hasattr(result, "original_text")
        assert hasattr(result, "decomposed_claim")
        assert hasattr(result, "atomic_verdicts")
        assert hasattr(result, "logic_aggregation")
        assert hasattr(result, "final_verdict")
        assert hasattr(result, "metadata")


class TestDelphiJuryAggregation:
    """Tests for Delphi Jury logic."""

    def test_jury_agreement_calculation(self) -> None:
        """Test calculation of jury agreement rate."""
        verdicts = [
            {"verdict": "true", "confidence": 0.8},
            {"verdict": "true", "confidence": 0.9},
            {"verdict": "false", "confidence": 0.6},
        ]

        # Calculate agreement with most common verdict
        verdict_counts: dict[str, int] = {}
        for v in verdicts:
            verdict = v["verdict"]
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        most_common = max(verdict_counts.items(), key=lambda x: x[1])
        agreement = most_common[1] / len(verdicts)

        assert agreement == 2 / 3
        assert most_common[0] == "true"

    def test_confidence_aggregation(self) -> None:
        """Test confidence aggregation across jury members."""
        verdicts = [
            {"verdict": "true", "confidence": 0.8},
            {"verdict": "true", "confidence": 0.9},
            {"verdict": "true", "confidence": 0.85},
        ]

        # Average confidence weighted by agreement
        avg_confidence = sum(v["confidence"] for v in verdicts) / len(verdicts)

        assert avg_confidence == pytest.approx(0.85)
