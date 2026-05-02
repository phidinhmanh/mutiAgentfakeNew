"""Shared parsing helpers for TRUST agent outputs."""

from __future__ import annotations

from typing import Any

from trust_agents.utils import clean_and_parse_json


def normalize_message_content(content: Any) -> str:
    """Normalize LangChain/LangGraph message content into plain text."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if isinstance(content, str):
        return content
    return str(content)


def extract_last_message_text(messages: list[Any]) -> str:
    """Extract normalized text from the last message in a message list."""
    if not messages:
        return ""
    return normalize_message_content(getattr(messages[-1], "content", ""))


def extract_json_object(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract a JSON object or list from free-form text."""
    parsed = clean_and_parse_json(text)
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def parse_claims_payload(text: str) -> list[str]:
    """Parse a claims payload into a clean list of claim strings."""
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        return []

    claims = parsed.get("claims", [])
    if not isinstance(claims, list):
        return []

    cleaned_claims: list[str] = []
    for claim in claims:
        if not claim:
            continue
        value = str(claim).strip()
        if value:
            cleaned_claims.append(value)
    return cleaned_claims


def parse_evidence_payload(text: str) -> list[dict[str, Any]]:
    """Parse an evidence payload into a list of evidence items."""
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        return []

    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list):
        return []

    result = []
    for item in evidence:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, str) and item.strip():
            result.append({"text": item.strip(), "source": "web", "score": 0.8})

    return result


def parse_dict_payload(text: str) -> dict[str, Any] | None:
    """Parse a dict payload from free-form text."""
    parsed = extract_json_object(text)
    if isinstance(parsed, dict):
        return parsed
    return None
