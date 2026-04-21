# -*- coding: utf-8 -*-
"""LangChain-compatible Gemini chat model supporting Google AI Studio and NVIDIA NIM.

This module provides a LangChain BaseChatModel implementation that wraps
Google Gemini models, supporting both direct Google AI Studio API and
NVIDIA NIM gateway.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import google.genai as genai
from google.genai import types
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, Field


class ChatGemini(BaseChatModel):
    """LangChain-compatible Chat model for Google Gemini.

    Supports:
    - Google AI Studio API (default)
    - NVIDIA NIM gateway (when provider="nvidia")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.1)
    max_output_tokens: int = Field(default=2048)
    provider: str = Field(default="google")  # "google" or "nvidia"
    google_api_key: str | None = None
    nvidia_api_key: str | None = None

    def _initialize_client(self) -> None:
        """Initialize the Gemini client with appropriate credentials."""
        if self.provider == "nvidia" and self.nvidia_api_key:
            genai.configure(api_key=self.nvidia_api_key)
        elif self.google_api_key:
            genai.configure(api_key=self.google_api_key)
        else:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY environment variable required"
                )
            genai.configure(api_key=api_key)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _format_messages(self, messages: list[BaseMessage]) -> str:
        """Convert LangChain messages to Gemini prompt format."""
        parts = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"Assistant: {msg.content}")

        return "\n\n".join(parts)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation."""
        self._initialize_client()

        prompt = self._format_messages(messages)
        model = genai.GenerativeModel(self.model_name)

        generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            stop_sequences=stop,
        )

        response = model.generate_content(prompt, generation_config=generation_config)
        text = response.text if hasattr(response, "text") else str(response)

        ai_message = AIMessage(content=text)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation - delegates to sync version."""
        import asyncio

        return await asyncio.to_thread(self._generate, messages, stop, **kwargs)

    def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[ChatGeneration]:
        """Streaming generation."""
        self._initialize_client()

        prompt = self._format_messages(messages)
        model = genai.GenerativeModel(self.model_name)

        generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        response = model.generate_content(prompt, generation_config=generation_config)

        for chunk in response:
            if chunk.text:
                yield ChatGeneration(message=AIMessage(content=chunk.text))


def create_chat_gemini(
    model_name: str = "gemini-2.0-flash",
    provider: str = "google",
    temperature: float = 0.1,
    **kwargs: Any,
) -> ChatGemini:
    """Factory function to create ChatGemini instance.

    Args:
        model_name: Gemini model to use (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
        provider: "google" for AI Studio, "nvidia" for NVIDIA NIM
        temperature: Sampling temperature
        **kwargs: Additional arguments passed to ChatGemini

    Returns:
        ChatGemini instance
    """
    return ChatGemini(
        model_name=model_name,
        provider=provider,
        temperature=temperature,
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        **kwargs,
    )