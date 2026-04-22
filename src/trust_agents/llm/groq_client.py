# -*- coding: utf-8 -*-
"""Groq LLM Client for TRUST Agents.

Supports Groq's free-tier LLM API (Llama-3, Mixtral, etc.)
with very fast inference and generous rate limits.
"""

from __future__ import annotations

import os
from typing import Any

from groq import Groq
from pydantic import BaseModel, Field


class GroqConfig(BaseModel):
    """Configuration for Groq client."""

    model: str = Field(default="llama-3.3-70b-versatile")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=2048)


class GroqClient:
    """Groq LLM client with streaming support."""

    def __init__(self, config: GroqConfig | None = None) -> None:
        self.config = config or GroqConfig()
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Synchronous generation."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": kwargs.get("system_prompt", "")},
                {"role": "user", "content": prompt},
            ],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, prompt: str, **kwargs: Any) -> Any:
        """Streaming generation."""
        return self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": kwargs.get("system_prompt", "")},
                {"role": "user", "content": prompt},
            ],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
        )


def create_groq_client(
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
) -> GroqClient:
    """Factory function to create Groq client."""
    config = GroqConfig(model=model, temperature=temperature)
    return GroqClient(config)