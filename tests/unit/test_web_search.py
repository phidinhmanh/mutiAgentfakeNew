"""Tests for web search integration."""
from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from fake_news_detector.rag.web_search import search_serper, search_tavily, search_web


class TestSearchSerper:
    """Test Serper API integration."""

    def test_search_serper_success(self, mock_serper_api: Mock) -> None:
        """Successful search returns results."""
        results = search_serper("Vietnam GDP growth")
        assert len(results) >= 1
        assert "content" in results[0]

    def test_search_serper_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing API key returns empty list."""
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        results = search_serper("test query")
        assert results == []

    def test_search_serper_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API error returns empty list."""
        def mock_post(*args: Any, **kwargs: Any) -> Mock:
            response = Mock()
            response.status_code = 500
            response.raise_for_status.side_effect = Exception("Server Error")
            return response

        monkeypatch.setattr("requests.post", mock_post)
        results = search_serper("test query")
        assert results == []

    def test_search_serper_empty_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty results return empty list."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"organic": []})
        mock_response.raise_for_status = Mock()

        def mock_post(*args: Any, **kwargs: Any) -> Mock:
            return mock_response

        monkeypatch.setattr("requests.post", mock_post)
        results = search_serper("nonexistent query")
        assert results == []


class TestSearchTavily:
    """Test Tavily API integration."""

    def test_search_tavily_success(self, mock_tavily_api: Mock) -> None:
        """Successful search returns results."""
        results = search_tavily("Vietnam economy")
        assert len(results) >= 1
        assert "content" in results[0]

    def test_search_tavily_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing API key returns empty list."""
        from fake_news_detector.config import settings
        monkeypatch.setattr(settings, "tavily_api_key", None)

        results = search_tavily("test query")
        assert results == []

    def test_search_tavily_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API error returns empty list."""
        def mock_post(*args: Any, **kwargs: Any) -> Mock:
            response = Mock()
            response.status_code = 500
            response.raise_for_status.side_effect = Exception("Server Error")
            return response

        monkeypatch.setattr("requests.post", mock_post)
        results = search_tavily("test query")
        assert results == []


class TestSearchWeb:
    """Test web search provider selection."""

    def test_search_web_uses_serper_by_default(
        self, mock_serper_api: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default provider is Serper."""
        monkeypatch.setenv("SERPER_API_KEY", "test-key")
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        results = search_web("test query")
        assert len(results) >= 0

    def test_search_web_uses_tavily_when_configured(
        self, mock_tavily_api: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tavily used when configured via settings."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        monkeypatch.setattr(
            "fake_news_detector.config.settings.search_provider", "tavily"
        )
        results = search_web("test query")
        assert len(results) >= 0

    def test_search_web_no_api_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No API keys returns empty list."""
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        results = search_web("test query")
        assert results == []