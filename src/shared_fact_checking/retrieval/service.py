"""Shared retrieval orchestration service."""
from __future__ import annotations

import logging
from typing import Any, Callable

from shared_fact_checking.retrieval.policy import (
    calculate_confidence_score,
    merge_results,
)

logger = logging.getLogger(__name__)


def retrieve_with_fallback(
    query: str,
    local_search: Callable[[str], list[dict[str, Any]]],
    web_search: Callable[[str], list[dict[str, Any]]],
    threshold: float,
    use_web_search: bool = True,
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Run local retrieval first, then fallback to web search if confidence is low."""
    local_results = local_search(query)
    logger.info("Local retrieval returned %d results", len(local_results))

    confidence = calculate_confidence_score(local_results)
    logger.info("Local retrieval confidence: %.3f, threshold: %.3f", confidence, threshold)

    if use_web_search and confidence < threshold:
        logger.info("Confidence below threshold, using web search")
        web_results = web_search(query)
        logger.info("Web search returned %d results", len(web_results))
        return merge_results(local_results, web_results, max_results=max_results)

    return local_results[:max_results]
