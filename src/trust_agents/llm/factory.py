"""Centralized factory for TRUST chat models."""
from __future__ import annotations

import os
from typing import Any

from trust_agents.config import LLMConfig, get_llm_config

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - dependency may be absent in some environments
    ChatOpenAI = None  # type: ignore[assignment]

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ImportError:  # pragma: no cover - dependency may be absent
    ChatNVIDIA = None  # type: ignore[assignment]

# Lazy import: google-genai triggers pydantic-core DLL load on Windows.
# Only import when the google provider is actually used.
_ChatGemini: type | None = None


def _get_chat_gemini() -> type:
    """Lazily import ChatGemini to avoid loading google-genai + pydantic prematurely."""
    global _ChatGemini
    if _ChatGemini is None:
        from trust_agents.llm.gemini_langchain import ChatGemini

        _ChatGemini = ChatGemini
    return _ChatGemini


def create_chat_model(config: LLMConfig | None = None) -> Any:
    """Create a chat model from TRUST configuration."""
    resolved_config = config or get_llm_config()

    provider_val = resolved_config.provider.value
    if provider_val == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is required for OpenAI provider")
        return ChatOpenAI(
            model=resolved_config.model,
            temperature=resolved_config.temperature,
            openai_api_key=resolved_config.get_api_key(),
        )

    if provider_val == "groq":
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is required for Groq provider")
        return ChatOpenAI(
            model=resolved_config.model,
            temperature=resolved_config.temperature,
            openai_api_key=resolved_config.get_api_key(),
            base_url="https://api.groq.com/openai/v1",
        )

    if provider_val == "nvidia":
        nvidia_key = os.getenv("NVIDIA_API_KEY") or resolved_config.get_api_key()
        if not nvidia_key:
            raise ValueError("NVIDIA_API_KEY environment variable required")
        if ChatNVIDIA is not None:
            return ChatNVIDIA(
                model=resolved_config.model,
                temperature=resolved_config.temperature,
                nvidia_api_key=nvidia_key,
                max_completion_tokens=resolved_config.max_tokens,
            )
        ChatGemini_cls = _get_chat_gemini()
        return ChatGemini_cls(
            model_name=resolved_config.model,
            provider="nvidia",
            temperature=resolved_config.temperature,
            nvidia_api_key=nvidia_key,
            google_api_key=None,
        )

    ChatGemini_cls = _get_chat_gemini()
    return ChatGemini_cls(
        model_name=resolved_config.model,
        provider=provider_val,
        temperature=resolved_config.temperature,
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
    )
