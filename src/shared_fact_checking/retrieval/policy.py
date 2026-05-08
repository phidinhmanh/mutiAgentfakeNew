"""Shared retrieval policy helpers."""

from __future__ import annotations

from typing import Any


def calculate_confidence_score(results: list[dict[str, Any]]) -> float:
    """Calculate confidence score for retrieval results."""
    if not results:
        return 0.0

    scores = [result.get("score", 0.0) for result in results]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    return (avg_score + max_score) / 2


def merge_results(
    local_results: list[dict[str, Any]],
    web_results: list[dict[str, Any]],
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Merge local and web retrieval results while deduplicating content."""
    seen_contents: set[str] = set()
    merged: list[dict[str, Any]] = []

    for result in local_results:
        content = result.get("content", result.get("text", ""))[:200]
        if content not in seen_contents:
            seen_contents.add(content)
            merged_result = result.copy()
            merged_result["source"] = merged_result.get("source", "vi_fact_check")
            merged.append(merged_result)

    for result in web_results:
        content = result.get("content", result.get("text", ""))[:200]
        if content not in seen_contents:
            seen_contents.add(content)
            merged_result = result.copy()
            merged_result["source"] = merged_result.get("source", "web_search")
            merged.append(merged_result)

    merged.sort(key=lambda item: item.get("score", 0), reverse=True)
    return merged[:max_results]
