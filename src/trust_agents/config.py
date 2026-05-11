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


DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.GEMINI_GOOGLE: "gemini-2.0-flash",
    LLMProvider.GEMINI_NVIDIA: "qwen/qwen3.5-122b-a10b",
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
}

MODEL_ENV_VARS = {
    LLMProvider.OPENAI: "OPENAI_MODEL",
    LLMProvider.GEMINI_GOOGLE: "GEMINI_MODEL",
    LLMProvider.GEMINI_NVIDIA: "NVIDIA_MODEL",
    LLMProvider.GROQ: "GROQ_MODEL",
}

PROVIDER_ENV_VARS = {
    LLMProvider.OPENAI: ("OPENAI_API_KEY",),
    LLMProvider.GEMINI_GOOGLE: ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    LLMProvider.GEMINI_NVIDIA: ("NVIDIA_API_KEY",),
    LLMProvider.GROQ: ("GROQ_API_KEY", "GROQ_KEY"),
}


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
        provider = LLMProvider(os.getenv("LLM_PROVIDER", "google").lower())
        return cls.from_provider(provider)

    @classmethod
    def from_provider(cls, provider: LLMProvider) -> LLMConfig:
        """Build provider config using provider-specific env overrides."""
        model_env = MODEL_ENV_VARS[provider]
        return cls(
            provider=provider,
            model=os.getenv(model_env, DEFAULT_MODELS[provider]),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )

    def get_api_key(self) -> str | None:
        """Get API key for current provider."""
        for env_name in PROVIDER_ENV_VARS[self.provider]:
            if api_key := os.getenv(env_name):
                return api_key
        return None

    def model_copy(self) -> LLMConfig:
        """Return a shallow copy (compatible with existing code)."""
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


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


def get_auto_llm_config() -> LLMConfig:
    """Auto-select the best available provider based on API keys."""
    # Priority: NVIDIA (NIM) > OpenAI > Gemini > Groq
    priority = [
        LLMProvider.GEMINI_NVIDIA,
        LLMProvider.OPENAI,
        LLMProvider.GEMINI_GOOGLE,
        LLMProvider.GROQ,
    ]

    for provider in priority:
        config = LLMConfig.from_provider(provider)
        if config.get_api_key():
            return config

    # Fallback to default
    return get_llm_config()
