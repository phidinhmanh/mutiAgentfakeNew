# -*- coding: utf-8 -*-
"""LangChain-compatible Gemini chat model supporting Google AI Studio and NVIDIA NIM."""

from __future__ import annotations

import os
from typing import Any, Iterator

from google import genai
from google.genai import types
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr


class ChatGemini(BaseChatModel):
    """LangChain-compatible chat model for Gemini providers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.1)
    max_output_tokens: int = Field(default=2048)
    provider: str = Field(default="google")
    google_api_key: str | None = None
    nvidia_api_key: str | None = None
    _client: genai.Client | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client

        if self.provider == "nvidia":
            api_key = self.nvidia_api_key or os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY environment variable required")
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    base_url="https://integrate.api.nvidia.com"
                ),
            )
            return self._client

        api_key = (
            self.google_api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable required"
            )

        self._client = genai.Client(api_key=api_key)
        return self._client

    def _build_config(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get(
                "max_output_tokens",
                kwargs.get("max_tokens", self.max_output_tokens),
            ),
            stop_sequences=stop,
        )

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text
        return str(response)

    def _format_messages(self, messages: list[BaseMessage]) -> str:
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"Assistant: {msg.content}")
            else:
                parts.append(str(msg.content))
        return "\n\n".join(parts)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._format_messages(messages)
        response = self._get_client().models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._build_config(stop=stop, **kwargs),
        )
        generation = ChatGeneration(message=AIMessage(content=self._extract_text(response)))
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        import asyncio

        return await asyncio.to_thread(self._generate, messages, stop, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        prompt = self._format_messages(messages)
        response = self._get_client().models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=self._build_config(stop=stop, **kwargs),
        )

        for chunk in response:
            text = self._extract_text(chunk)
            if text:
                yield ChatGeneration(message=AIMessage(content=text))

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ChatGemini":
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "ChatGemini":
        return self


def create_chat_gemini(
    model_name: str = "gemini-2.0-flash",
    provider: str = "google",
    temperature: float = 0.1,
    **kwargs: Any,
) -> ChatGemini:
    """Factory function to create ChatGemini instance."""
    return ChatGemini(
        model_name=model_name,
        provider=provider,
        temperature=temperature,
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        **kwargs,
    )
