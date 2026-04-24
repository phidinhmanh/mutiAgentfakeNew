# -*- coding: utf-8 -*-
"""LLM Configuration for TRUST Agents.

Provides unified configuration for LLM backends supporting:
- OpenAI (default in news_agent)
- Google Gemini (via Google AI Studio)
- Google Gemini (via NVIDIA NIM)
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI_GOOGLE = "google"
    GEMINI_NVIDIA = "nvidia"
    GROQ = "groq"


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""

    provider: LLMProvider = Field(default=LLMProvider.GEMINI_GOOGLE)
    model: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=2048)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        provider_str = os.getenv("LLM_PROVIDER", "google").lower()
        provider = LLMProvider(provider_str)

        # Map provider string to model defaults
        model_map = {
            LLMProvider.OPENAI: os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            LLMProvider.GEMINI_GOOGLE: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            LLMProvider.GEMINI_NVIDIA: os.getenv("GEMINI_MODEL", "google/gemma-4-26b-a4b-it"),
            LLMProvider.GROQ: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        }

        return cls(
            provider=provider,
            model=model_map.get(provider, "gemini-2.0-flash"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )

    def get_api_key(self) -> str | None:
        """Get API key for current provider."""
        if self.provider == LLMProvider.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        elif self.provider in (LLMProvider.GEMINI_GOOGLE, LLMProvider.GEMINI_NVIDIA):
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif self.provider == LLMProvider.GROQ:
            return os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")
        return None


# Global config instance
_config: LLMConfig | None = None


def get_llm_config() -> LLMConfig:
    """Get or create global LLM config."""
    global _config
    if _config is None:
        _config = LLMConfig.from_env()
    return _config


def set_llm_config(config: LLMConfig) -> None:
    """Set global LLM config."""
    global _config
    _config = config