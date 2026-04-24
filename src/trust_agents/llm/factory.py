"""Centralized factory for TRUST chat models."""
from __future__ import annotations

import os
from typing import Any

from trust_agents.config import LLMConfig, get_llm_config
from trust_agents.llm.gemini_langchain import ChatGemini

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - dependency may be absent in some environments
    ChatOpenAI = None  # type: ignore[assignment]


def create_chat_model(config: LLMConfig | None = None) -> Any:
    """Create a chat model from TRUST configuration."""
    resolved_config = config or get_llm_config()

    if resolved_config.provider.value == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is required for OpenAI provider")
        return ChatOpenAI(
            model=resolved_config.model,
            temperature=resolved_config.temperature,
            openai_api_key=resolved_config.get_api_key(),
        )

    return ChatGemini(
        model_name=resolved_config.model,
        provider=resolved_config.provider.value,
        temperature=resolved_config.temperature,
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
    )
