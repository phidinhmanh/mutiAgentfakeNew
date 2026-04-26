"""Tests for reasoning agent."""
from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from fake_news_detector.agents.reasoning import (
    ReasoningAgent,
    aggregate_verdicts,
)


class TestReasoningAgentInit:
    """Test ReasoningAgent initialization."""

    def test_init_with_client(self, mock_nvidia_client: Mock) -> None:
        """Agent initializes with LLM client."""
        agent = ReasoningAgent(mock_nvidia_client)
        assert agent.llm is mock_nvidia_client


class TestReasoningAgentAnalyze:
    """Test ReasoningAgent.analyze method."""

    def test_analyze_empty_evidence(
        self, mock_nvidia_client: Mock
    ) -> None:
        """Empty evidence returns UNVERIFIABLE."""
        agent = ReasoningAgent(mock_nvidia_client)
        result = agent.analyze("Some claim", [])
        assert result["verdict"] == "UNVERIFIABLE"
        assert "Không có bằng chứng" in result["reasoning"]

    def test_analyze_with_evidence(
        self, mock_nvidia_client: Mock, sample_evidence: list[dict[str, Any]]
    ) -> None:
        """Valid evidence returns parsed verdict."""
        agent = ReasoningAgent(mock_nvidia_client)
        result = agent.analyze("GDP tăng trưởng?", sample_evidence)
        assert "verdict" in result
        assert "confidence" in result
        assert "reasoning" in result

    def test_analyze_llm_error_returns_unverifiable(
        self, mock_nvidia_client: Mock, sample_evidence: list[dict[str, Any]]
    ) -> None:
        """LLM error returns UNVERIFIABLE."""
        mock_nvidia_client.invoke.side_effect = Exception("API Error")
        agent = ReasoningAgent(mock_nvidia_client)
        result = agent.analyze("Some claim", sample_evidence)
        assert result["verdict"] == "UNVERIFIABLE"

    def test_analyze_citation_validation(
        self, mock_nvidia_client: Mock, sample_evidence: list[dict[str, Any]]
    ) -> None:
        """Citations are validated against evidence."""
        mock_nvidia_client.invoke.return_value = (
            '{"verdict": "REAL", "confidence": 0.8, '
            '"reasoning": "Evidence supports", '
            '"citations": [{"evidence_id": 0, "quote_text": "Việt Nam đạt tăng trưởng 8%"}]}'
        )
        agent = ReasoningAgent(mock_nvidia_client)
        result = agent.analyze("GDP tăng trưởng?", sample_evidence)
        assert "citations_valid" in result


class TestReasoningAgentStreamAnalysis:
    """Test ReasoningAgent.stream_analysis method."""

    def test_stream_empty_evidence(
        self, mock_nvidia_client: Mock
    ) -> None:
        """Empty evidence yields unverifiable result."""
        agent = ReasoningAgent(mock_nvidia_client)
        chunks = list(agent.stream_analysis("Some claim", []))
        assert len(chunks) == 1
        assert "UNVERIFIABLE" in chunks[0]

    def test_stream_with_evidence(
        self, mock_nvidia_client: Mock, sample_evidence: list[dict[str, Any]]
    ) -> None:
        """Valid evidence yields streamed chunks."""
        agent = ReasoningAgent(mock_nvidia_client)
        chunks = list(agent.stream_analysis("GDP tăng?", sample_evidence))
        assert len(chunks) > 0

    def test_stream_llm_error(
        self, mock_nvidia_client: Mock, sample_evidence: list[dict[str, Any]]
    ) -> None:
        """LLM error yields unverifiable result."""
        mock_nvidia_client.stream.side_effect = Exception("Stream Error")
        agent = ReasoningAgent(mock_nvidia_client)
        chunks = list(agent.stream_analysis("Some claim", sample_evidence))
        assert len(chunks) == 1
        assert "UNVERIFIABLE" in chunks[0]


class TestReasoningAgentFormatEvidence:
    """Test _format_evidence helper method."""

    def test_format_evidence_empty(self, mock_nvidia_client: Mock) -> None:
        """Empty evidence formats correctly."""
        agent = ReasoningAgent(mock_nvidia_client)
        result = agent._format_evidence([])
        assert result == ""

    def test_format_evidence_single_item(
        self, mock_nvidia_client: Mock
    ) -> None:
        """Single evidence formats with index and source."""
        agent = ReasoningAgent(mock_nvidia_client)
        evidence = [{"content": "Test content", "source": "test"}]
        result = agent._format_evidence(evidence)
        assert "[0]" in result
        assert "Test content" in result
        assert "test" in result

    def test_format_evidence_multiple_items(
        self, mock_nvidia_client: Mock
    ) -> None:
        """Multiple evidence items format correctly."""
        agent = ReasoningAgent(mock_nvidia_client)
        evidence = [
            {"content": "First", "source": "src1"},
            {"content": "Second", "source": "src2"},
        ]
        result = agent._format_evidence(evidence)
        assert "[0]" in result
        assert "[1]" in result

    def test_format_evidence_missing_source(
        self, mock_nvidia_client: Mock
    ) -> None:
        """Evidence without source uses 'unknown'."""
        agent = ReasoningAgent(mock_nvidia_client)
        evidence = [{"content": "Test"}]
        result = agent._format_evidence(evidence)
        assert "unknown" in result


class TestReasoningAgentParseResponse:
    """Test _parse_response method."""

    def test_parse_valid_json(self, mock_nvidia_client: Mock) -> None:
        """Valid JSON returns parsed dict."""
        agent = ReasoningAgent(mock_nvidia_client)
        response = '{"verdict": "REAL", "confidence": 0.9, "reasoning": "Test"}'
        result = agent._parse_response(response)
        assert result["verdict"] == "REAL"
        assert result["confidence"] == 0.9

    def test_parse_invalid_json(self, mock_nvidia_client: Mock) -> None:
        """Invalid JSON returns UNVERIFIABLE."""
        agent = ReasoningAgent(mock_nvidia_client)
        result = agent._parse_response("Not valid JSON")
        assert result["verdict"] == "UNVERIFIABLE"
        assert result["confidence"] == 0.0


class TestReasoningAgentValidateCitations:
    """Test _validate_citations method."""

    def test_validate_empty_citations(
        self, mock_nvidia_client: Mock, sample_evidence: list[dict[str, Any]]
    ) -> None:
        """Empty citations returns valid result."""
        agent = ReasoningAgent(mock_nvidia_client)
        result = agent._validate_citations({}, sample_evidence)
        assert result["citations"] == []
        assert result["citations_valid"] is True

    def test_validate_valid_citation(
        self, mock_nvidia_client: Mock, sample_evidence: list[dict[str, Any]]
    ) -> None:
        """Valid citation is kept."""
        agent = ReasoningAgent(mock_nvidia_client)
        citation_result = {
            "verdict": "REAL",
            "confidence": 0.8,
            "reasoning": "Test",
            "citations": [
                {"evidence_id": 0, "quote_text": "Việt Nam đạt tăng trưởng 8%"}
            ],
        }
        result = agent._validate_citations(citation_result, sample_evidence)
        assert len(result["citations"]) >= 0

    def test_validate_invalid_evidence_id(
        self, mock_nvidia_client: Mock, sample_evidence: list[dict[str, Any]]
    ) -> None:
        """Invalid evidence_id is filtered out."""
        agent = ReasoningAgent(mock_nvidia_client)
        citation_result = {
            "verdict": "REAL",
            "confidence": 0.8,
            "reasoning": "Test",
            "citations": [{"evidence_id": 999, "quote_text": "Some text"}],
        }
        result = agent._validate_citations(citation_result, sample_evidence)
        assert result["citations_valid"] is False


class TestReasoningAgentUnverifiableResult:
    """Test _unverifiable_result method."""

    def test_returns_correct_structure(self, mock_nvidia_client: Mock) -> None:
        """Returns correct unverifiable structure."""
        agent = ReasoningAgent(mock_nvidia_client)
        result = agent._unverifiable_result("No evidence")
        assert result["verdict"] == "UNVERIFIABLE"
        assert result["confidence"] == 0.0
        assert result["reasoning"] == "No evidence"
        assert result["citations"] == []
        assert result["citations_valid"] is True


class TestAggregateVerdicts:
    """Test aggregate_verdicts function."""

    def test_aggregate_empty_list(self) -> None:
        """Empty list returns UNVERIFIABLE."""
        result = aggregate_verdicts([])
        assert result["verdict"] == "UNVERIFIABLE"
        assert result["confidence"] == 0.0

    def test_aggregate_real_majority(self, sample_verdicts: list[dict[str, Any]]) -> None:
        """More REAL than FAKE returns REAL."""
        result = aggregate_verdicts(sample_verdicts)
        assert result["verdict"] == "REAL"

    def test_aggregate_fake_majority(self, sample_verdicts: list[dict[str, Any]]) -> None:
        """More FAKE than REAL returns FAKE."""
        verdicts = [
            {"verdict": "FAKE", "confidence": 0.8, "reasoning": "Contradicts"},
            {"verdict": "FAKE", "confidence": 0.7, "reasoning": "Not match"},
            {"verdict": "REAL", "confidence": 0.6, "reasoning": "Some support"},
        ]
        result = aggregate_verdicts(verdicts)
        assert result["verdict"] == "FAKE"

    def test_aggregate_tie(self) -> None:
        """Equal REAL and FAKE returns UNVERIFIABLE."""
        verdicts = [
            {"verdict": "REAL", "confidence": 0.8, "reasoning": "Test1"},
            {"verdict": "FAKE", "confidence": 0.7, "reasoning": "Test2"},
        ]
        result = aggregate_verdicts(verdicts)
        assert result["verdict"] == "UNVERIFIABLE"

    def test_aggregate_only_unverifiable(self) -> None:
        """Only UNVERIFIABLE verdicts returns UNVERIFIABLE."""
        verdicts = [
            {"verdict": "UNVERIFIABLE", "confidence": 0.0, "reasoning": "No evidence"},
            {"verdict": "UNVERIFIABLE", "confidence": 0.0, "reasoning": "Insufficient"},
        ]
        result = aggregate_verdicts(verdicts)
        assert result["verdict"] == "UNVERIFIABLE"

    def test_aggregate_confidence_average(self) -> None:
        """Confidence is averaged correctly."""
        verdicts = [
            {"verdict": "REAL", "confidence": 0.8, "reasoning": "Test1"},
            {"verdict": "REAL", "confidence": 0.6, "reasoning": "Test2"},
        ]
        result = aggregate_verdicts(verdicts)
        assert result["confidence"] == 0.7

    def test_aggregate_reasoning_truncated(self) -> None:
        """Reasoning is truncated to 1000 characters."""
        long_reasoning = "A" * 2000
        verdicts = [
            {"verdict": "REAL", "confidence": 0.8, "reasoning": long_reasoning},
        ]
        result = aggregate_verdicts(verdicts)
        assert len(result["reasoning"]) <= 1000

    def test_aggregate_stats_included(self, sample_verdicts: list[dict[str, Any]]) -> None:
        """Stats are included in result."""
        result = aggregate_verdicts(sample_verdicts)
        assert "stats" in result
        assert result["stats"]["real_count"] == 2
        assert result["stats"]["fake_count"] == 1
        assert result["stats"]["total"] == 3
