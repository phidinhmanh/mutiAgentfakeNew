"""LLM Helper functions for TRUST Agents tools.

Provides a unified interface for LLM calls across all tool files.
Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

from trust_agents.config import get_llm_config

load_dotenv()
logger = logging.getLogger("trust_agents.llm_helpers")


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
) -> str:
    """Call LLM with prompt, supporting multiple backends."""
    config = get_llm_config()

    if config.provider.value == "openai":
        from openai import OpenAI

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = OpenAI(api_key=config.get_api_key())
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    if config.provider.value == "groq":
        from groq import Groq

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = Groq(api_key=config.get_api_key())
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    if config.provider.value == "nvidia":
        from openai import OpenAI

        api_key = os.getenv("NVIDIA_API_KEY") or config.get_api_key()
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("A string NVIDIA API key is required")

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    from google import genai
    from google.genai import types

    api_key = config.get_api_key()
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("A string Gemini API key is required")

    client = genai.Client(api_key=api_key)
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    response = client.models.generate_content(
        model=config.model,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    return response.text.strip() if hasattr(response, "text") else str(response)


def _clean_json_text(text: str) -> str:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    cleaned = re.sub(r"^[^{\[]*", "", cleaned, count=1, flags=re.DOTALL)
    cleaned = re.sub(r"[^}\]]*$", "", cleaned, count=1, flags=re.DOTALL)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned.strip()


def _extract_balanced_json(text: str) -> str | None:
    start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    if not start_positions:
        return None

    start = min(start_positions)
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _parse_json_response(response: str) -> dict[str, Any]:
    candidates = [response]

    cleaned = _clean_json_text(response)
    if cleaned and cleaned not in candidates:
        candidates.append(cleaned)

    balanced = _extract_balanced_json(cleaned or response)
    if balanced and balanced not in candidates:
        candidates.append(balanced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("LLM returned JSON that is not an object")

    raise ValueError(f"Could not parse LLM response as JSON: {response[:200]}")


def call_llm_json(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
) -> dict[str, Any]:
    """Call LLM and parse response as JSON."""
    response = call_llm(prompt, system_prompt, temperature, max_tokens)

    try:
        return _parse_json_response(response)
    except ValueError as first_error:
        logger.warning("Failed to parse JSON on first attempt: %s", first_error)

    retry_prompt = (
        f"{prompt}\n\n"
        "Return exactly one valid JSON object. Do not include markdown fences, "
        "commentary, or trailing text."
    )
    retry_response = call_llm(retry_prompt, system_prompt, temperature, max_tokens)

    try:
        return _parse_json_response(retry_response)
    except ValueError as error:
        logger.warning(
            "Failed to parse JSON after retry: %s, response: %s",
            error,
            retry_response[:200],
        )
        raise
