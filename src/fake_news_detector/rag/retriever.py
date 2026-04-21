"""Retriever for hybrid RAG (FAISS + web search)."""
import logging
from typing import Any

from fake_news_detector.config import settings
from fake_news_detector.rag.vector_store import get_vector_store
from fake_news_detector.rag.web_search import search_web

logger = logging.getLogger(__name__)


def calculate_confidence_score(results: list[dict[str, Any]]) -> float:
    """Calculate confidence score for retrieval results.

    Args:
        results: List of retrieved documents

    Returns:
        Confidence score between 0 and 1
    """
    if not results:
        return 0.0

    scores = [r.get("score", 0.0) for r in results]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)

    return (avg_score + max_score) / 2


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
    faiss_results = vector_store.similarity_search(
        claim, k=settings.top_k_faiss
    )

    logger.info(f"FAISS returned {len(faiss_results)} results")

    confidence = calculate_confidence_score(faiss_results)
    logger.info(f"FAISS confidence: {confidence:.3f}, threshold: {settings.similarity_threshold}")

    if use_web_search and confidence < settings.similarity_threshold:
        logger.info("Confidence below threshold, using web search...")
        web_results = search_web(claim, num_results=settings.top_k_google)
        logger.info(f"Web search returned {len(web_results)} results")
        return _merge_results(faiss_results, web_results)

    return faiss_results


def _merge_results(
    faiss_results: list[dict[str, Any]],
    web_results: list[dict[str, Any]],
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Merge FAISS and web search results.

    Args:
        faiss_results: Results from FAISS
        web_results: Results from web search
        max_results: Maximum number of results to return

    Returns:
        Merged and sorted results
    """
    seen_contents = set()
    merged = []

    for result in faiss_results:
        content = result.get("content", "")[:200]
        if content not in seen_contents:
            seen_contents.add(content)
            result["source"] = result.get("source", "vi_fact_check")
            merged.append(result)

    for result in web_results:
        content = result.get("content", "")[:200]
        if content not in seen_contents:
            seen_contents.add(content)
            result["source"] = result.get("source", "web_search")
            merged.append(result)

    merged.sort(key=lambda x: x.get("score", 0), reverse=True)
    return merged[:max_results]