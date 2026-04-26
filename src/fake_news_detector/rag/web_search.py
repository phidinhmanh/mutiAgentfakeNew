"""Web search integration for real-time evidence."""

import logging
from typing import Any

import requests
from requests import HTTPError

from fake_news_detector.config import settings

logger = logging.getLogger(__name__)


def search_serper(
    query: str, num_results: int = 3
) -> tuple[list[dict[str, Any]], str | None]:
    """Search using Serper.dev (Google Search API).

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        Tuple of (results, failure_reason)
    """
    if not settings.serper_api_key:
        logger.warning("SERPER_API_KEY not set, skipping search")
        return [], "missing_api_key"

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": num_results}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("organic", [])[:num_results]:
            results.append(
                {
                    "content": item.get("snippet", ""),
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "source": "serper",
                    "score": 0.8,
                }
            )

        return results, None
    except HTTPError as error:
        status_code = error.response.status_code if error.response is not None else None
        if status_code == 403:
            logger.error(
                "Serper search failed with 403 Forbidden; check SERPER_API_KEY or provider access"
            )
            return [], "http_403"
        logger.error("Serper search failed with HTTP %s: %s", status_code, error)
        return [], f"http_{status_code}" if status_code is not None else "http_error"
    except Exception as error:
        logger.error("Serper search failed: %s", error)
        return [], type(error).__name__


def _should_fallback_to_tavily(
    failure_reason: str | None, results: list[dict[str, Any]]
) -> bool:
    """Return whether Tavily should be used after Serper attempt."""
    if results:
        return False
    if not settings.tavily_api_key:
        return False
    return failure_reason is not None


def _log_tavily_fallback_reason(failure_reason: str | None) -> None:
    """Log why Tavily fallback is being used."""
    if failure_reason == "http_403":
        logger.warning("Falling back to Tavily because Serper returned 403 Forbidden")
    elif failure_reason == "missing_api_key":
        logger.warning("Falling back to Tavily because SERPER_API_KEY is unavailable")
    elif failure_reason:
        logger.warning(
            "Falling back to Tavily because Serper failed: %s", failure_reason
        )


def search_web(query: str, num_results: int = 3) -> list[dict[str, Any]]:
    """Search the web using configured provider.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        List of search results
    """
    if settings.search_provider == "tavily":
        return search_tavily(query, num_results)

    results, failure_reason = search_serper(query, num_results)
    if results:
        return results

    if not _should_fallback_to_tavily(failure_reason, results):
        return []

    _log_tavily_fallback_reason(failure_reason)
    return search_tavily(query, num_results)


def search_tavily(query: str, num_results: int = 3) -> list[dict[str, Any]]:
    """Search using Tavily AI Search API.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        List of search results
    """
    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY not set, skipping search")
        return []

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": num_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", [])[:num_results]:
            results.append(
                {
                    "content": item.get("content", ""),
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": "tavily",
                    "score": 0.8,
                }
            )

        return results
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []


def check_semantic_consistency(
    article: str,
    claim: str,
    trusted_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Check semantic consistency with trusted sources.

    Args:
        article: Article text
        claim: Claim to check
        trusted_sources: List of trusted domain names

    Returns:
        Dictionary with consistency check results
    """
    if trusted_sources is None:
        trusted_sources = settings.trusted_sources

    inconsistencies = []
    entities_article = _extract_key_entities(article)

    for source in trusted_sources[:2]:
        search_query = f"{claim} site:{source}"
        results = search_web(search_query, num_results=2)

        if results:
            entities_trusted = _extract_key_entities(results[0].get("content", ""))
            diff = entities_article - entities_trusted
            if diff:
                inconsistencies.append(
                    {
                        "source": source,
                        "diff": list(diff),
                    }
                )

    return {
        "consistent": len(inconsistencies) == 0,
        "inconsistencies": inconsistencies,
    }


def _extract_key_entities(text: str) -> set[str]:
    """Extract key entities from text."""
    import re

    words = re.findall(r"\b[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*\b", text)
    return set(words[:20])
