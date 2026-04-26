"""Retriever for hybrid RAG (FAISS + web search)."""

import logging
from typing import Any

from fake_news_detector.config import settings
from fake_news_detector.rag.vector_store import get_vector_store
from fake_news_detector.rag.web_search import search_web
from shared_fact_checking.retrieval.policy import (
    merge_results,
)
from shared_fact_checking.retrieval.service import retrieve_with_fallback

logger = logging.getLogger(__name__)


def retrieve_evidence(claim: str, use_web_search: bool = True) -> list[dict[str, Any]]:
    """Retrieve evidence using hybrid approach.

    Agent 2 workflow:
    1. Search FAISS first
    2. If confidence is weak, search Google
    3. Merge results

    Args:
        claim: Claim to retrieve evidence for
        use_web_search: Whether to use web search as fallback

    Returns:
        List of evidence documents with scores
    """
    vector_store = get_vector_store()
    return retrieve_with_fallback(
        query=claim,
        local_search=lambda value: vector_store.similarity_search(
            value, k=settings.top_k_faiss
        ),
        web_search=lambda value: search_web(value, num_results=settings.top_k_google),
        threshold=settings.similarity_threshold,
        use_web_search=use_web_search,
    )


def _merge_results(
    faiss_results: list[dict[str, Any]],
    web_results: list[dict[str, Any]],
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for shared merge logic."""
    return merge_results(faiss_results, web_results, max_results=max_results)
