"""LLM Configuration for TRUST Agents.

Provides unified configuration for LLM backends supporting:
- OpenAI (default in news_agent)
- Google Gemini (via Google AI Studio)
- Google Gemini (via NVIDIA NIM)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI_GOOGLE = "google"
    GEMINI_NVIDIA = "nvidia"
    GROQ = "groq"


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""

    provider: LLMProvider = field(default=LLMProvider.GEMINI_GOOGLE)
    model: str = field(default="gemini-2.0-flash")
    temperature: float = field(default=0.1)
    max_tokens: int = field(default=2048)

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Load configuration from environment variables."""
        provider_str = os.getenv("LLM_PROVIDER", "google").lower()
        provider = LLMProvider(provider_str)

        model_map = {
            LLMProvider.OPENAI: os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            LLMProvider.GEMINI_GOOGLE: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            LLMProvider.GEMINI_NVIDIA: os.getenv(
                "NVIDIA_MODEL", "qwen/qwen3.5-122b-a10b"
            ),
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
        elif self.provider == LLMProvider.GEMINI_GOOGLE:
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif self.provider == LLMProvider.GEMINI_NVIDIA:
            return os.getenv("NVIDIA_API_KEY")
        elif self.provider == LLMProvider.GROQ:
            return os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")
        return None

    def model_copy(self) -> LLMConfig:
        """Return a shallow copy (compatible with existing code)."""
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


def get_llm_config() -> LLMConfig:
    """Get LLM config from environment."""
    return LLMConfig.from_env()
