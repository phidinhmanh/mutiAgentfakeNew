"""Shared LLM utilities: sync wrappers, JSON parsing, and model creation."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_async_in_thread(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine in a background thread pool.

    Reusable replacement for the 4 duplicate sync-wrapper functions in each agent.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    def _run() -> T:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run).result()


def parse_llm_json_response(response: Any) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks.

    Shared replacement for the duplicate parsing block in claim_extractor,
    verifier, and explainer agents.
    """
    content = getattr(response, "content", str(response))
    content = content.strip()
    content = re.sub(r"```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```\s*", "", content).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[^}]+(?:\{[^}]*\}[^}]*)*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}


def get_domain_authority(url: str) -> str:
    """Return authority level ('high', 'medium', 'low') for a given URL.

    Checks against AUTHORITATIVE_DOMAINS. Government domains get 'high'.
    Major news sites get 'high'. Others get 'medium'.
    """
    from shared_fact_checking.constants import AUTHORITATIVE_DOMAINS

    url_lower = url.lower()
    if any(d in url_lower for d in (".gov.vn", "chinhphu.vn", "baochinhphu.vn")):
        return "high"
    if any(d in url_lower for d in AUTHORITATIVE_DOMAINS):
        return "high"
    return "medium"
