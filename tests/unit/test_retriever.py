"""Tests for hybrid retriever."""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from fake_news_detector.rag.retriever import (
    retrieve_evidence,
)
from shared_fact_checking.retrieval.policy import (
    calculate_confidence_score,
    merge_results as _merge_results,
)
from shared_fact_checking.retrieval.service import _retrieval_cache


@pytest.fixture(autouse=True)
def clear_retrieval_cache():
    """Clear the global retrieval cache before each test to ensure isolation."""
    _retrieval_cache.clear()


class TestCalculateConfidenceScore:
    """Test calculate_confidence_score function."""

    def test_calculate_empty_results(self) -> None:
        """Empty list returns 0.0."""
        result = calculate_confidence_score([])
        assert result == 0.0

    def test_calculate_single_result(self) -> None:
        """Single result returns its score."""
        results = [{"score": 0.9}]
        result = calculate_confidence_score(results)
        assert result == 0.9

    def test_calculate_average_plus_max(self) -> None:
        """Score = (avg + max) / 2."""
        results = [{"score": 0.6}, {"score": 0.8}, {"score": 0.7}]
        result = calculate_confidence_score(results)
        avg = (0.6 + 0.8 + 0.7) / 3
        max_score = 0.8
        expected = (avg + max_score) / 2
        assert result == expected

    def test_calculate_missing_score_uses_zero(self) -> None:
        """Missing score defaults to 0.0."""
        results = [{"content": "test"}]
        result = calculate_confidence_score(results)
        assert result == 0.0


class TestMergeResults:
    """Test _merge_results function."""

    def test_merge_empty_inputs(self) -> None:
        """Empty inputs return empty list."""
        result = _merge_results([], [])
        assert result == []

    def test_merge_only_faiss(self) -> None:
        """Only FAISS results returned."""
        faiss = [{"content": "FAISS result", "score": 0.9}]
        result = _merge_results(faiss, [])
        assert len(result) == 1
        assert result[0]["content"] == "FAISS result"

    def test_merge_only_web(self) -> None:
        """Only web results returned."""
        web = [{"content": "Web result", "score": 0.8}]
        result = _merge_results([], web)
        assert len(result) == 1

    def test_merge_no_duplicate_urls(self) -> None:
        """No duplicate content in merged result."""
        faiss = [{"content": "Same content here", "score": 0.9, "url": "http://test"}]
        web = [{"content": "Same content here", "score": 0.8, "url": "http://test"}]
        result = _merge_results(faiss, web)
        assert len(result) == 1

    def test_merge_sorts_by_score(self) -> None:
        """Results sorted by score descending."""
        faiss = [{"content": "Lower", "score": 0.5}]
        web = [{"content": "Higher", "score": 0.9}]
        result = _merge_results(faiss, web)
        assert result[0]["content"] == "Higher"

    def test_merge_max_results(self) -> None:
        """Results limited to max_results."""
        faiss = [{"content": f"FAISS {i}", "score": 0.9 - i * 0.01} for i in range(5)]
        web = [{"content": f"Web {i}", "score": 0.8 - i * 0.01} for i in range(5)]
        result = _merge_results(faiss, web, max_results=3)
        assert len(result) == 3

    def test_merge_sets_source(self) -> None:
        """Source field is set on results."""
        faiss = [{"content": "test", "score": 0.9}]
        web = [{"content": "web test", "score": 0.8}]
        result = _merge_results(faiss, web)
        sources = [r.get("source") for r in result]
        assert "vi_fact_check" in sources
        assert "web_search" in sources


class TestRetrieveEvidence:
    """Test retrieve_evidence function."""

    def test_retrieve_empty_claim(self) -> None:
        """Empty claim returns empty list."""
        with patch("fake_news_detector.rag.retriever.get_vector_store") as mock_get:
            mock_vs = Mock()
            mock_vs.similarity_search.return_value = []
            mock_get.return_value = mock_vs

            result = retrieve_evidence("")
            assert result == []

    @patch("fake_news_detector.rag.retriever.get_vector_store")
    @patch("fake_news_detector.rag.retriever.search_web")
    def test_retrieve_faiss_only_high_confidence(
        self, mock_search: Mock, mock_get: Mock
    ) -> None:
        """High confidence skips web search."""
        mock_vs = Mock()
        mock_vs.similarity_search.return_value = [{"content": "high", "score": 0.95}]
        mock_get.return_value = mock_vs

        result = retrieve_evidence("test claim")
        assert len(result) == 1
        mock_search.assert_not_called()

    @patch("fake_news_detector.rag.retriever.get_vector_store")
    @patch("fake_news_detector.rag.retriever.search_web")
    def test_retrieve_web_fallback_low_confidence(
        self, mock_search: Mock, mock_get: Mock
    ) -> None:
        """Low confidence triggers web search."""
        mock_vs = Mock()
        mock_vs.similarity_search.return_value = [{"content": "low", "score": 0.3}]
        mock_get.return_value = mock_vs
        mock_search.return_value = [{"content": "web result", "score": 0.8}]

        result = retrieve_evidence("test claim", use_web_search=True)
        mock_search.assert_called_once()
        assert len(result) == 2

    @patch("fake_news_detector.rag.retriever.get_vector_store")
    def test_retrieve_web_search_disabled(
        self, mock_get: Mock
    ) -> None:
        """use_web_search=False skips web search."""
        mock_vs = Mock()
        mock_vs.similarity_search.return_value = [{"content": "low", "score": 0.3}]
        mock_get.return_value = mock_vs

        result = retrieve_evidence("test claim", use_web_search=False)
        assert len(result) == 1
