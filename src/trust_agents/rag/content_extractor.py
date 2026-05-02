"""Content Extraction Module - Bypasses web restrictions, extracts & truncates content.

Converts HTML to Markdown using trafilatura for better LLM comprehension.
Implements Header Rotation (User-Agent) and Smart Token Truncation.
"""

import asyncio
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

# Random User-Agent rotation pool (desktop browsers)
_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 "
    "Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]


def _get_random_ua() -> str:
    """Return a random User-Agent to avoid 403 blocks."""
    return random.choice(_UA_POOL)


def extract_from_url(url: str, timeout: int = 10) -> str:
    """Extract main content from a URL as Markdown.

    Args:
        url: URL to scrape
        timeout: Request timeout in seconds

    Returns:
        Markdown content or empty string on failure
    """
    try:
        trafilatura = _lazy_import_trafilatura()
        ua = _get_random_ua()
        downloaded = None

        # Use requests as primary fetcher for better header/SSL control
        import requests
        try:
            headers = {"User-Agent": ua}
            # Disable SSL verification for maximum compatibility (dangerous but common in scrapers)
            response = requests.get(url, headers=headers, timeout=timeout, verify=False)
            if response.status_code == 200:
                downloaded = response.text
            else:
                logger.debug(f"Requests returned {response.status_code} for {url}")
        except Exception as req_e:
            logger.debug(f"Requests failed for {url}: {req_e}")

        # Fallback to trafilatura's fetcher if requests fails
        if not downloaded:
            try:
                # Note: trafilatura.fetch_url might not support user_agent kwarg in all versions
                downloaded = trafilatura.fetch_url(url, no_ssl=True)
            except Exception as traf_e:
                logger.debug(f"Trafilatura fetch failed for {url}: {traf_e}")

        if not downloaded:
            logger.warning(f"Failed to download content from {url}")
            return ""

        # Convert HTML → Markdown (preserves headings, lists, structure)
        markdown = trafilatura.extract(downloaded, output_format="markdown")
        if markdown:
            logger.info(f"Extracted {len(markdown)} chars from {url[:60]}...")
            return markdown.strip()

        logger.warning(f"Trafilatura returned empty for {url}")
        return ""

    except Exception as e:
        logger.warning(f"Failed to extract {url}: {e}")
        return ""


# Blacklisted domains that are slow, require JS, or often fail
_BLACKLISTED_DOMAINS = [
    "youtube.com", "facebook.com", "twitter.com", "t.co", "instagram.com",
    "bnews.vn", "zalo.me", "zalo.com",
    # Educational sites that often return irrelevant content (e.g. math problems)
    "vndoc.com", "vietjack.com", "loigiaihay.com", "giaitoan.com",
    "cunghocvui.com", "lazi.vn", "olm.vn", "khoahoc.vietjack.com",
    "tech12h.com", "hoc24h.vn", "moon.vn",
]


def _remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics for case-insensitive matching."""
    import unicodedata
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _score_content_relevance(text: str, query: str) -> float:
    """Score how relevant extracted content is to the query.

    Returns a score 0.0-1.0 based on keyword overlap and content density.
    """
    if not text or not query:
        return 0.5  # neutral

    text_norm = _remove_diacritics(text).lower()
    query_norm = _remove_diacritics(query).lower()

    # Extract key terms from query: proper nouns, numbers, important words
    key_terms = []
    # Numbers (years, dates, percentages)
    key_terms.extend(re.findall(r"\b\d{4}\b", query))  # years
    key_terms.extend(re.findall(r"\b\d+(?:[.,]\d+)*%?\b", query))  # numbers
    # Proper nouns (Vietnamese capitalized) - more inclusive
    key_terms.extend(re.findall(r"\b[A-ZÀ-Ỹ][a-zà-ỹ0-9]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ0-9]+){0,3}\b", query))
    # Important words (3+ chars) from normalized query
    key_terms.extend(re.findall(r"\b\w{3,}\b", query_norm))

    if not key_terms:
        # Fallback: split query into individual words
        key_terms = [w for w in query_norm.split() if len(w) >= 2]

    if not key_terms:
        return 0.5

    # Count how many key terms appear in text (case/diacritic insensitive)
    # Deduplicate key terms to avoid over-weighting repeated words in query
    unique_terms = {_remove_diacritics(t).lower() for t in key_terms}
    matches = sum(1 for term in unique_terms if term in text_norm)
    ratio = matches / len(unique_terms) if unique_terms else 0.5

    # Penalty for very short content (likely failed extraction)
    if len(text.strip()) < 200:
        ratio *= 0.5

    # Penalty for content density (if too much boilerplate)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        avg_line_len = sum(len(line) for line in lines) / len(lines)
        if avg_line_len < 30:  # likely menu/nav, not article
            ratio *= 0.7

    return min(1.0, ratio)


async def extract_content_batch_async(
    urls: list[str],
    query: str = "",
    max_chars: int = 8000,
    max_results: int | None = None,
    concurrency: int = 5,
) -> list[dict[str, Any]]:
    """Extract content from multiple URLs in parallel.

    Args:
        urls: List of URLs to scrape
        query: Optional query for smart truncation
        max_chars: Max chars per result
        max_results: Maximum number of successful results to return
        concurrency: Max parallel requests

    Returns:
        List of {url, title, content_markdown} dicts
    """
    if max_results is None:
        max_results = len(urls)

    # Filter blacklisted domains
    filtered_urls = [
        url for url in urls
        if not any(domain in url.lower() for domain in _BLACKLISTED_DOMAINS)
    ]

    if not filtered_urls:
        return []

    # Use ThreadPoolExecutor for blocking trafilatura calls
    loop = asyncio.get_running_loop()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        tasks = []
        for url in filtered_urls[:max_results + 2]:
            tasks.append(loop.run_in_executor(executor, extract_from_url, url))

        extracted_contents = await asyncio.gather(*tasks, return_exceptions=True)

    # Score and filter by relevance, sort by score descending
    scored_results = []
    for url, content in zip(filtered_urls, extracted_contents, strict=True):
        if content and isinstance(content, str) and len(content.strip()) > 100:
            truncated = smart_truncate(content, query, max_chars)
            score = _score_content_relevance(truncated, query)
            scored_results.append((score, url, truncated))

    # Filter by minimum relevance threshold and sort
    min_relevance = 0.15
    scored_results = [
        (s, u, c) for s, u, c in scored_results if s >= min_relevance
    ]
    scored_results.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, url, truncated in scored_results:
        results.append({
            "url": url,
            "title": _extract_title_from_markdown(truncated),
            "content": truncated,
            "source": _extract_domain(url),
            "relevance_score": round(score, 3),
        })
        if len(results) >= max_results:
            break

    logger.info(
        f"Extracted {len(results)}/{len(urls)} URLs "
        f"(filtered by relevance >={min_relevance})"
    )
    return results


def smart_truncate(text: str, query: str = "", max_chars: int = 8000) -> str:
    """Truncate text intelligently around query keywords.

    Keeps the most relevant chunk (~max_chars chars) that contains
    query keywords. Falls back to first max_chars chars.

    Args:
        text: Input text (markdown)
        query: Keywords to find (claim or query text)
        max_chars: Maximum characters to keep

    Returns:
        Truncated text, preserving structure where possible
    """
    if len(text) <= max_chars:
        return text

    if not query:
        # No query: just take the beginning
        return text[:max_chars]

    # Find the best position (most keyword matches nearby)
    keywords = re.findall(r"\b\w{4,}\b", query.lower())
    best_pos = 0
    best_score = 0

    # Scan through text in windows
    text_norm = _remove_diacritics(text).lower()
    keywords = re.findall(r"\b\w{4,}\b", _remove_diacritics(query).lower())

    window = max_chars
    step = 500
    for i in range(0, len(text) - window, step):
        window_text = text_norm[i : i + window]
        score = sum(1 for kw in keywords if kw in window_text)

        # Position bonus: prefer the beginning of the article (title/intro)
        if i == 0:
            score *= 1.5  # Heavy bonus for the start
        elif i < 2000:
            score *= 1.2

        if score > best_score:
            best_score = score
            best_pos = i

    # Extract window and trim to sentence boundary
    chunk = text[best_pos : best_pos + max_chars]

    # Find last sentence boundary to avoid cutting mid-sentence
    # Look for common sentence endings: . ! ? or newline after markdown heading
    last_cut = chunk.rfind("\n\n")  # paragraph break
    if last_cut > max_chars * 0.5:  # only cut at paragraph if not too early
        chunk = chunk[:last_cut]

    logger.debug(f"Smart truncate: kept {len(chunk)} chars (score={best_score})")
    return chunk


def extract_content_batch(
    urls: list[str],
    query: str = "",
    max_chars: int = 8000,
    delay: float = 1.0,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    """Extract content from multiple URLs with rate limiting and truncation.

    Args:
        urls: List of URLs to scrape
        query: Optional query for smart truncation
        max_chars: Max chars per result
        delay: Delay between requests (rate limit protection)
        max_results: Maximum number of successful results to return

    Returns:
        List of {url, title, content_markdown} dicts
    """
    if max_results is None:
        max_results = len(urls)

    results = []
    for url in urls:
        if len(results) >= max_results:
            break

        logger.info(f"Scraping: {url[:70]}...")
        content = extract_from_url(url)

        if content:
            truncated = smart_truncate(content, query, max_chars)
            results.append({
                "url": url,
                "title": _extract_title_from_markdown(truncated),
                "content": truncated,
                "source": _extract_domain(url),
            })
            logger.info(f"  -> Got {len(truncated)} chars")
        else:
            logger.warning("  -> Failed to extract content")

        time.sleep(delay)

    logger.info(f"Extracted content from {len(results)}/{len(urls)} URLs")
    return results


def _extract_title_from_markdown(text: str) -> str:
    """Extract the first H1 heading from markdown as a title."""
    if not text:
        return ""
    # Match # Heading or #Heading (with optional space)
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Fallback: first non-empty line
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped[:100]
    return ""


def _extract_domain(url: str) -> str:
    """Extract domain from URL for source attribution."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return url


# ---------------------------------------------------------------------------
# trafilatura is the primary extractor; lazy import to avoid DLL issues
# ---------------------------------------------------------------------------
def _lazy_import_trafilatura():
    try:
        import trafilatura
        return trafilatura
    except ImportError:
        logger.error("trafilatura not installed: run `uv add trafilatura`")
        raise
