from __future__ import annotations

from unittest.mock import Mock

from trust_agents.config import LLMConfig, LLMProvider
from trust_agents.llm.factory import create_chat_model


class TestCreateChatModel:
    """Test centralized TRUST model factory behavior."""

    def test_create_openai_model(self, monkeypatch) -> None:
        openai_ctor = Mock(return_value="openai-model")
        monkeypatch.setattr("trust_agents.llm.factory.ChatOpenAI", openai_ctor)
        monkeypatch.setattr("trust_agents.llm.factory.ChatGemini", Mock())

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            temperature=0.2,
        )

        result = create_chat_model(config)

        assert result == "openai-model"
        openai_ctor.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=config.get_api_key(),
        )

    def test_create_gemini_model(self, monkeypatch) -> None:
        gemini_ctor = Mock(return_value="gemini-model")
        monkeypatch.setattr("trust_agents.llm.factory.ChatOpenAI", Mock())
        monkeypatch.setattr("trust_agents.llm.factory.ChatGemini", gemini_ctor)
        monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")

        config = LLMConfig(
            provider=LLMProvider.GEMINI_GOOGLE,
            model="gemini-2.0-flash",
            temperature=0.1,
        )

        result = create_chat_model(config)

        assert result == "gemini-model"
        gemini_ctor.assert_called_once_with(
            model_name="gemini-2.0-flash",
            provider="google",
            temperature=0.1,
            google_api_key="gem-key",
            nvidia_api_key="nv-key",
        )
