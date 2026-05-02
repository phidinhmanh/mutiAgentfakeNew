"""Web search integration for real-time evidence."""

import logging
import re
import time
from functools import lru_cache
from typing import Any

import requests
from requests import HTTPError

from trust_agents.config import settings

logger = logging.getLogger(__name__)

# Domain preference scoring for Vietnamese news content
# Higher score = more likely to have quality news content
_NEWS_DOMAIN_SCORES: dict[str, float] = {
    # High-quality Vietnamese news sites
    "vnexpress.net": 1.0,
    "thanhnien.vn": 1.0,
    "tuoitre.vn": 1.0,
    "vietnamnet.vn": 1.0,
    "nhandan.vn": 0.95,
    "laodong.vn": 0.9,
    "qdnd.vn": 0.9,
    "vov.vn": 0.85,
    "vtc.vn": 0.85,
    "vietnamplus.vn": 0.85,
    "baotintuc.vn": 0.85,
    "baochinhphu.vn": 0.8,
    "hanoimoi.vn": 0.8,
    "dn.24h.com.vn": 0.75,
    "24h.com.vn": 0.7,
    # International news with VN coverage
    "bbc.com": 0.7,
    "reuters.com": 0.7,
    "apnews.com": 0.65,
    "scmp.com": 0.65,
    # Low-value domains to deprioritize
    "vietjack.com": -1.0,
    "vndoc.com": -1.0,
    "loigiaihay.com": -1.0,
    "giaitoan.com": -1.0,
    "cunghocvui.com": -1.0,
    "lazi.vn": -1.0,
    "vietnews.vn": -0.5,
}


def _score_domain(url: str) -> float:
    """Score a URL based on its domain for news relevance."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower().replace("www.", "")
        return _NEWS_DOMAIN_SCORES.get(domain, 0.0)
    except Exception:
        return 0.0


def _sort_by_domain_preference(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort search results by domain preference (news sites first)."""
    scored = []
    for r in results:
        url = r.get("url", "")
        score = _score_domain(url)
        scored.append((score, url, r))

    # Sort descending (highest score first), keep stable order for same score
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, _, r in scored]


# ---------------------------------------------------------------------------
# Query Simplification for Retry - extract keywords/numbers for fallback search
# ---------------------------------------------------------------------------
def simplify_query(query: str) -> str | None:
    """Simplify a query by keeping only nouns, proper nouns, and numbers.

    This is used for retry when the original long query fails.
    E.g., "Thủ tướng Việt Nam ký quyết định quan trọng" -> "Thủ tướng quyết định"
    Or "Mbappe ở tuổi 24 ghi 40 bàn" -> "Mbappe 24 40"

    Returns:
        Simplified query string, or None if query is already simple
    """
    if len(query) < 30:
        return None  # Already simple enough

    # Extract:
    # 1. Numbers (including dates, percentages, years)
    # 2. Proper nouns (Vietnamese capitalized words)
    # 3. Specific nouns

    numbers = re.findall(r"\b\d+(?:[.,]\d+)*%?\b", query)
    # Match capitalized words that could be names, places, organizations
    proper_nouns = re.findall(r"\b[A-ZÀ-Ỹ][a-zà-ỹ]*(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]*)*\b", query)

    # Vietnamese stopwords to remove
    stopwords = {
        "đã", "đang", "sẽ", "có", "được", "là", "và", "của", "cho", "với",
        "theo", "trong", "năm", "tháng", "ngày", "giờ", "phút",
        "này", "đó", "kia", "ở", "tại", "về", "ra", "vào", "từ",
        "những", "các", "một", "hai", "ba", "bốn", "năm", "sáu",
        "hôm", "nay", "qua", "đến", "bởi", "vì", "nên", "để",
        "tuy", "nhưng", "mà", "hay", "hoặc", "vẫn", "còn", "chỉ",
        "rằng", "khi", "nếu", "thì", "vì", "do", "vì",
        "cũng", "đều", "tất", "cả", "mọi", "ai", "gì", "đâu",
        "hiện", "mới", "vừa", "lại", "luôn", "mãi", "bao", "giờ",
        "như", "vậy", "thế", "nào", "sao", "hơn", "kém", "nhất",
    }

    # Filter proper nouns (remove stopwords)
    keywords = []
    for noun in proper_nouns:
        noun_lower = noun.lower()
        if noun_lower not in stopwords and len(noun) > 2:
            keywords.append(noun)

    # Combine: numbers first, then keywords
    parts = []

    # Add most important numbers (years, percentages, large numbers)
    for num in numbers:
        # Skip small numbers that are likely quantities in sentences
        if re.match(r"^\d{4}$", num):  # Years
            parts.append(num)
        elif "000" in num or len(num) > 4:  # Large numbers
            parts.append(num)
        elif "%" in num:  # Percentages
            parts.append(num)

    # Add keywords (max 5)
    parts.extend(keywords[:5])

    if not parts:
        return None

    simplified = " ".join(parts)
    if simplified.lower() == query.lower()[:len(simplified)].lower():
        return None  # No significant simplification

    logger.info(f"Query simplified: '{query[:50]}...' -> '{simplified[:50]}...'")
    return simplified


# ---------------------------------------------------------------------------
# Search Result Cache - avoids re-querying the same query
# ---------------------------------------------------------------------------
_search_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
_CACHE_TTL = 300  # 5 minutes


def _cached_search_results(
    query: str, num_results: int
) -> list[dict[str, Any]]:
    """Return cached results for a query if still fresh, else None."""
    cache_key = f"{query[:100]}:{num_results}"
    if cache_key in _search_cache:
        results, cached_at = _search_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL:
            return results
    return None


def _add_to_cache(query: str, num_results: int, results: list[dict[str, Any]]) -> None:
    """Cache search results for a query."""
    cache_key = f"{query[:100]}:{num_results}"
    _search_cache[cache_key] = (results, time.time())


def clear_search_cache() -> None:
    """Clear all cached search results."""
    _search_cache.clear()


# ---------------------------------------------------------------------------
# Serper Search
# ---------------------------------------------------------------------------
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
        response = requests.post(url, json=payload, headers=headers, timeout=30)
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


# ---------------------------------------------------------------------------
# Main Search Function
# ---------------------------------------------------------------------------
def search_web(
    query: str, num_results: int = 3, use_cache: bool = True
) -> list[dict[str, Any]]:
    """Search the web using configured provider with caching.

    Priority (for reliability and cost):
    1. Cache check (avoids redundant API calls)
    2. DuckDuckGo (free, no rate limits if using library) - try first
    3. Retry with simplified query if first attempt fails
    4. Tavily (paid, richer results) - try if DDG fails and available
    5. Serper (paid, Google-backed) - try if others fail

    Args:
        query: Search query
        num_results: Number of results to return
        use_cache: Whether to use/search result cache

    Returns:
        List of search results
    """
    # Check cache first
    if use_cache:
        cached = _cached_search_results(query, num_results)
        if cached is not None:
            logger.debug(f"Search cache HIT for: {query[:50]}...")
            return cached

    # Always try DuckDuckGo first (free, no rate limits)
    results = search_ddg(query, num_results)
    if results:
        _add_to_cache(query, num_results, results)
        return results

    # DuckDuckGo failed; skip HTML scrape timeout delays
    # Try site-specific search directly (faster than retrying DDG)
    vn_sites = ["vnexpress.net", "thanhnien.vn", "tuoitre.vn", "nhandan.vn"]
    for site in vn_sites[:2]:  # Try top 2 sites only
        vn_query = f"{query} site:{site}"
        logger.info(f"Fallback site search: '{vn_query[:50]}...'")
        results = search_ddg(vn_query, num_results)
        if results:
            _add_to_cache(query, num_results, results)
            return results

    # DuckDuckGo still failed; try paid providers
    # If Tavily is configured as primary, use it directly
    if settings.search_provider == "tavily" and settings.tavily_api_key:
        results = search_tavily(query, num_results)
        if results:
            _add_to_cache(query, num_results, results)
            return results

    # Try Serper as secondary
    results, failure_reason = search_serper(query, num_results)
    if results:
        _add_to_cache(query, num_results, results)
        return results

    # Serper failed; try Tavily as final fallback (if not already tried)
    if settings.tavily_api_key:
        _log_tavily_fallback_reason(failure_reason)
        results = search_tavily(query, num_results)
        if results:
            _add_to_cache(query, num_results, results)
            return results

    logger.warning(f"All search methods failed for query: {query[:50]}...")
    return []


# ---------------------------------------------------------------------------
# DuckDuckGo Search (URL extraction + snippet conversion)
# ---------------------------------------------------------------------------
def search_ddg_urls(query: str, num_results: int = 5) -> list[dict[str, Any]]:
    """Extract URLs from DuckDuckGo search results.

    Uses duckduckgo-search library with HTML scrape fallback.

    Args:
        query: Search query
        num_results: Number of URLs to return

    Returns:
        List of dicts with 'title', 'url', 'snippet', 'source'
    """
    results = []

    # Try HTML scrape first (fast, reliable when not blocked)
    # Uses curl_cffi for bot detection bypass
    try:
        import bs4
        from curl_cffi import requests as curl_requests
        from urllib.parse import quote, parse_qs, urlparse

        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"

        # Use curl_cffi to bypass DDG bot detection
        resp = curl_requests.get(
            url,
            impersonate="chrome",
            timeout=10,
            headers={
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )
        resp.raise_for_status()

        if len(resp.text) < 5000:
            logger.warning(
                f"DuckDuckGo HTML scrape returned short response "
                f"({len(resp.text)} bytes) - likely bot detection"
            )
        else:
            soup = bs4.BeautifulSoup(resp.text, "html.parser")
            for result in soup.select(".result")[:num_results]:
                a_tag = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if a_tag and a_tag.get("href"):
                    href = a_tag["href"]
                    if "uddg" in href:
                        parsed = urlparse(href)
                        params = parse_qs(parsed.query)
                        real_url = params.get("uddg", [""])[0]
                    else:
                        real_url = href

                    results.append({
                        "title": a_tag.get_text(strip=True),
                        "url": real_url,
                        "snippet": snippet.get_text(strip=True) if snippet else "",
                        "source": "duckduckgo",
                    })

            if results:
                logger.info(f"DuckDuckGo (HTML scrape) found {len(results)} URLs")
                # Sort by domain preference (news sites first)
                results = _sort_by_domain_preference(results)
                return results
    except Exception as e:
        logger.debug(f"DuckDuckGo HTML scrape failed or blocked: {e}")

    # Fallback to ddgs library (new name for duckduckgo_search)
    # Uses curl_cffi for better bot detection bypass
    try:
        from ddgs import DDGS

        with DDGS() as ddg:
            # Force region='vn-vi' and safesearch='off' for better Vietnamese results
            lib_results = list(ddg.text(
                query,
                region="vn-vi",
                safesearch="off",
                max_results=num_results
            ))
            for r in lib_results:
                href = r.get("href", "")
                if href and "duckduckgo" not in href:
                    results.append({
                        "title": r.get("title", ""),
                        "url": href,
                        "snippet": r.get("body", ""),
                        "source": "duckduckgo",
                    })
            if results:
                logger.info(f"DuckDuckGo (ddgs) found {len(results)} URLs")
                # Sort by domain preference (news sites first)
                results = _sort_by_domain_preference(results)
                return results
    except (Exception, Warning) as lib_err:
        logger.debug(f"ddgs library failed or blocked: {lib_err}")

    # Fallback: scrape DuckDuckGo HTML directly (always works)
    return _search_ddg_html(query, num_results)


def _search_ddg_html(query: str, num_results: int = 5) -> list[dict[str, Any]]:
    """Scrape DuckDuckGo HTML directly with curl_cffi for bot bypass.

    Uses curl_cffi to impersonate Chrome browser and bypass basic protection.
    """
    try:
        import bs4
        from curl_cffi import requests as curl_requests
        from urllib.parse import quote, parse_qs, urlparse

        # Use html.duckduckgo.com directly (bypasses redirect chain)
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"

        # Impersonate Chrome for better bot detection bypass
        resp = curl_requests.get(
            url,
            impersonate="chrome",
            timeout=10,
            headers={
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/avif,image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )
        resp.raise_for_status()

        # Validate response - short responses are usually bot detection pages
        # or timeout truncations
        if len(resp.text) < 5000:
            logger.warning(
                f"DuckDuckGo HTML scrape returned short response "
                f"({len(resp.text)} bytes) - likely bot detection"
            )
            return []

        soup = bs4.BeautifulSoup(resp.text, "html.parser")
        results = []

        for result in soup.select(".result")[:num_results]:
            a_tag = result.select_one(".result__a")
            snippet = result.select_one(".result__snippet")
            if a_tag and a_tag.get("href"):
                href = a_tag["href"]
                # Decode DuckDuckGo redirect URLs (//duckduckgo.com/l/?uddg=...)
                if "uddg" in href:
                    parsed = urlparse(href)
                    params = parse_qs(parsed.query)
                    real_url = params.get("uddg", [""])[0]
                else:
                    real_url = href

                results.append({
                    "title": a_tag.get_text(strip=True),
                    "url": real_url,
                    "snippet": snippet.get_text(strip=True) if snippet else "",
                    "source": "duckduckgo",
                })

        logger.info(f"DuckDuckGo (HTML scrape) found {len(results)} URLs")
        return results

    except Exception as e:
        logger.warning(f"DuckDuckGo HTML scrape failed: {e}")
        return []


def search_ddg(query: str, num_results: int = 3) -> list[dict[str, Any]]:
    """Search using DuckDuckGo (Free fallback) - returns snippets for backward compat."""
    # Use URLs first, then convert to legacy format
    urls = search_ddg_urls(query, num_results)
    return [
        {
            "content": r.get("snippet", ""),
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "source": "duckduckgo",
            "score": 0.6,
        }
        for r in urls
    ]


# ---------------------------------------------------------------------------
# Tavily Search
# ---------------------------------------------------------------------------
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
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": True,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
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


# ---------------------------------------------------------------------------
# Semantic Consistency Check
# ---------------------------------------------------------------------------
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