# -*- coding: utf-8 -*-
"""Unified Gemini LLM Client.

Supports both Google AI Studio and NVIDIA NIM backends for Gemini models.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Self

import google.genai as genai
from google.genai import types
from pydantic import BaseModel, Field


class GeminiConfig(BaseModel):
    """Configuration for Gemini client."""

    model: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.1)
    max_output_tokens: int = Field(default=2048)
    provider: str = Field(default="google")  # "google" or "nvidia"


class GeminiClient:
    """Unified Gemini client supporting Google AI Studio and NVIDIA NIM."""

    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or GeminiConfig()

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        nvidia_key = os.getenv("NVIDIA_API_KEY")

        if self.config.provider == "nvidia" and nvidia_key:
            genai.configure(api_key=nvidia_key)
            self._base_url = "https://integrate.api.nvidia.com/v1"
        else:
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY environment variable required"
                )
            genai.configure(api_key=api_key)
            self._base_url = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Synchronous generation."""
        response = self._create_model().generate_content(
            prompt,
            generation_config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=kwargs.get("max_tokens", self.config.max_output_tokens),
            ),
        )
        return response.text

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        """Async generation."""
        import asyncio
        return await asyncio.to_thread(self.generate, prompt, **kwargs)

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Streaming generation."""
        model = self._create_model()
        response = model.generate_content(
            prompt,
            generation_config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=kwargs.get("max_tokens", self.config.max_output_tokens),
            ),
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _create_model(self) -> Any:
        """Create model instance."""
        if self._base_url:
            return genai.GenerativeModel(
                self.config.model,
                vertex_url=self._base_url,
            )
        return genai.GenerativeModel(self.config.model)


def create_gemini_client(
    model: str = "gemini-2.0-flash",
    provider: str = "google",
    temperature: float = 0.1,
) -> GeminiClient:
    """Factory function to create Gemini client."""
    config = GeminiConfig(
        model=model,
        provider=provider,
        temperature=temperature,
    )
    return GeminiClient(config)