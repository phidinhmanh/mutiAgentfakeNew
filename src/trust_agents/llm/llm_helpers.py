# -*- coding: utf-8 -*-
"""LLM Helper functions for TRUST Agents tools.

Provides a unified interface for LLM calls across all tool files.
Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

from __future__ import annotations

import os
import json
import logging
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
    """
    Call LLM with prompt, supporting multiple backends.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        LLM response text
    """
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
        return response.choices[0].message.content.strip()

    else:
        # Use Gemini
        import google.genai as genai

        genai.configure(api_key=config.get_api_key())
        model = genai.GenerativeModel(config.model)

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        return response.text.strip() if hasattr(response, "text") else str(response)


def call_llm_json(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
) -> dict[str, Any]:
    """
    Call LLM and parse response as JSON.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        Parsed JSON response

    Raises:
        ValueError: If response cannot be parsed as JSON
    """
    response = call_llm(prompt, system_prompt, temperature, max_tokens)

    # Try to extract JSON from response
    try:
        # Remove markdown code blocks if present
        cleaned = response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}, response: {response[:200]}")
        raise ValueError(f"Could not parse LLM response as JSON: {response[:200]}")