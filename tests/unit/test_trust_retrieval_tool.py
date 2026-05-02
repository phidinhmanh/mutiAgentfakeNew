from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from trust_agents.agents.retrieval_agent_tools import (
    _merge_faiss_web,
    search_evidence_tool,
)


class TestMergeFaissWeb:
    """Test TRUST retrieval merge helper semantics."""

    def test_merge_deduplicates_by_truncated_content(self) -> None:
        faiss_results = [{"content": "same content", "score": 0.9}]
        web_results = [{"content": "same content", "score": 0.8}]

        result = _merge_faiss_web(faiss_results, web_results)

        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_merge_sets_default_sources(self) -> None:
        faiss_results = [{"content": "faiss", "score": 0.9}]
        web_results = [{"content": "web", "score": 0.8}]

        result = _merge_faiss_web(faiss_results, web_results)

        assert result[0]["source"] == "vi_fact_check"
        assert result[1]["source"] == "web_search"


class TestSearchEvidenceTool:
    """Test TRUST retrieval tool output contract."""

    @pytest.mark.asyncio
    async def test_returns_faiss_results_without_web_search_when_confident(self, monkeypatch) -> None:
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = [
            {"content": "Evidence A", "score": 0.95, "source": "dataset"}
        ]

        monkeypatch.setattr(
            "trust_agents.agents.retrieval_agent_tools._get_rag_components",
            lambda: (lambda: mock_vector_store, Mock()),
        )

        raw = await search_evidence_tool.ainvoke({"query": "claim text", "top_k": 3, "use_web_search": True})
        result = json.loads(raw)

        assert result["query"] == "claim text"
        assert result["top_k"] == 3
        assert result["total_found"] == 1
        assert result["evidence"][0]["text"] == "Evidence A"
        assert result["evidence"][0]["source"] == "dataset"

    @pytest.mark.asyncio
    async def test_adds_web_results_when_confidence_is_low(self, monkeypatch) -> None:
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = [{"content": "Low confidence", "score": 0.2}]
        mock_search_web = Mock(
            return_value=[
                {
                    "content": "Web evidence",
                    "score": 0.8,
                    "url": "https://example.com",
                }
            ]
        )

        monkeypatch.setattr(
            "trust_agents.agents.retrieval_agent_tools._get_rag_components",
            lambda: (lambda: mock_vector_store, Mock()),
        )
        # Patch at the original definition site. Since search_web is imported
        # via "from X import search_web" inside the tool function, the reference
        # captured at import time points to the module-level attribute, so patching
        # it at its original definition site intercepts the call regardless of where
        # it's imported.
        monkeypatch.setattr(
            "trust_agents.rag.web_search.search_web",
            mock_search_web,
        )

        raw = await search_evidence_tool.ainvoke({"query": "claim text", "top_k": 5, "use_web_search": True})
        result = json.loads(raw)

        assert result["total_found"] == 2
        assert any(item["text"] == "Web evidence" for item in result["evidence"])
        mock_search_web.assert_called_once_with("claim text", num_results=5)

    @pytest.mark.asyncio
    async def test_returns_error_payload_on_exception(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "trust_agents.agents.retrieval_agent_tools._get_rag_components",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        raw = await search_evidence_tool.ainvoke({"query": "claim text", "top_k": 5, "use_web_search": True})
        result = json.loads(raw)

        assert result["evidence"] == []
        assert result["query"] == "claim text"
        assert result["top_k"] == 5
        assert "boom" in result["error"]
