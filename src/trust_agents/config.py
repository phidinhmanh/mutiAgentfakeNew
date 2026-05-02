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
from typing import Literal

from dotenv import load_dotenv

# Explicitly load .env - override shell env vars so .env takes precedence
load_dotenv(override=True)


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
    model: str = field(default="gemma-4-26b-a4b-it")
    temperature: float = field(default=0.1)
    max_tokens: int = field(default=2048)

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Load configuration from environment variables."""
        provider_str = os.getenv("LLM_PROVIDER", "google").lower()
        provider = LLMProvider(provider_str)

        model_map = {
            LLMProvider.OPENAI: os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            LLMProvider.GEMINI_GOOGLE: os.getenv("GEMINI_MODEL", "gemma-4-31b-it"),
            LLMProvider.GEMINI_NVIDIA: os.getenv(
                "NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"
            ),
            LLMProvider.GROQ: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        }

        return cls(
            provider=provider,
            model=model_map.get(provider, "gemma-4-31b-it"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )

    def get_api_key_for(self, provider: LLMProvider | str) -> str | None:
        """Get API key for a specific provider."""
        provider_value = (
            provider.value if isinstance(provider, LLMProvider) else str(provider)
        )
        if provider_value == LLMProvider.OPENAI.value:
            return os.getenv("OPENAI_API_KEY")
        if provider_value == LLMProvider.GEMINI_GOOGLE.value:
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if provider_value == LLMProvider.GEMINI_NVIDIA.value:
            return os.getenv("NVIDIA_API_KEY")
        if provider_value == LLMProvider.GROQ.value:
            return os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")
        return None

    def get_api_key(self) -> str | None:
        """Get API key for current provider."""
        return self.get_api_key_for(self.provider)

    def model_copy(self) -> LLMConfig:
        """Return a shallow copy (compatible with existing code)."""
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


@dataclass
class Settings:
    """Application settings loaded from environment variables (Legacy Merge)."""

    # NVIDIA NIM API
    nvidia_api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))

    # Search APIs
    serper_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # HuggingFace
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    # Model settings
    phobert_model: str = field(
        default_factory=lambda: os.getenv("PHOBERT_MODEL", "vinai/phobert-base")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )

    # RAG settings
    faiss_index_path: str = field(
        default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    )
    top_k_faiss: int = field(default_factory=lambda: int(os.getenv("TOP_K_FAISS", "5")))
    top_k_google: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_GOOGLE", "3"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    )

    # Search engine choice
    search_provider: Literal["serper", "tavily"] = field(
        default_factory=lambda: os.getenv("SEARCH_PROVIDER", "serper")
    )  # type: ignore

    # Application settings
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true"
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048"))
    )
    streaming: bool = field(
        default_factory=lambda: os.getenv("STREAMING", "True").lower() == "true"
    )

    # Trusted Vietnamese news sources
    trusted_sources: list[str] = field(
        default_factory=lambda: [
            "vnexpress.net",
            "nhandan.vn",
            "baotintuc.vn",
            "thanhnien.vn",
            "tuoitre.vn",
        ]
    )

    # Cache settings
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))


settings = Settings()


def get_llm_config() -> LLMConfig:
    """Get LLM config from environment."""
    return LLMConfig.from_env()
