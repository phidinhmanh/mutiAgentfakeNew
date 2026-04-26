"""Shared retrieval orchestration service."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from shared_fact_checking.retrieval.policy import (
    calculate_confidence_score,
    merge_results,
)

logger = logging.getLogger(__name__)

# Bounded LRU cache for retrieval results — avoids unbounded memory growth.
# Key: normalized query text. Value: (final_results, confidence, local_results).
_CACHE_MAX_SIZE = 200
_retrieval_cache: OrderedDict[
    str, tuple[list[dict[str, Any]], float, list[dict[str, Any]]]
] = OrderedDict()


def _normalize_query_for_cache(query: str) -> str:
    """Normalize query for cache key — lowercase, strip whitespace."""
    return query.strip().lower()


def _cache_key(
    query: str, threshold: float, use_web_search: bool, max_results: int
) -> str:
    """Build a cache key from query and retrieval parameters."""
    normalized = _normalize_query_for_cache(query)
    return hashlib.md5(
        f"{normalized}|{threshold}|{use_web_search}|{max_results}".encode()
    ).hexdigest()


def retrieve_with_fallback(
    query: str,
    local_search: Callable[[str], list[dict[str, Any]]],
    web_search: Callable[[str], list[dict[str, Any]]],
    threshold: float,
    use_web_search: bool = True,
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Run local retrieval first, then fallback to web search if confidence is low.

    Results are cached in-process to avoid repeated local + web calls for the same query.
    """
    cache_key = _cache_key(query, threshold, use_web_search, max_results)

    # Check cache hit
    if cache_key in _retrieval_cache:
        logger.info("Retrieval cache hit for query: %s...", query[:40])
        _retrieval_cache.move_to_end(cache_key)
        results, _, _ = _retrieval_cache[cache_key]
        return results[:max_results]

    local_results = local_search(query)
    logger.info("Local retrieval returned %d results", len(local_results))

    confidence = calculate_confidence_score(local_results)
    logger.info(
        "Local retrieval confidence: %.3f, threshold: %.3f", confidence, threshold
    )

    if use_web_search and confidence < threshold:
        logger.info("Confidence below threshold, using web search")
        web_results = web_search(query)
        logger.info("Web search returned %d results", len(web_results))
        results = merge_results(local_results, web_results, max_results=max_results)
    else:
        results = local_results[:max_results]

    # Store in cache (raw local_results + confidence for reuse)
    _retrieval_cache[cache_key] = (results, confidence, local_results)
    _retrieval_cache.move_to_end(cache_key)

    # Evict oldest entry if cache grew beyond limit
    while len(_retrieval_cache) > _CACHE_MAX_SIZE:
        _retrieval_cache.popitem(last=False)

    return results
