from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.benchmark import (
    _score_prediction,
    create_benchmark_model,
    extract_multi_agent_verdict,
    extract_vifactcheck_sample,
    normalize_sample_label,
    parse_verdict_label,
)
from trust_agents.config import LLMConfig, LLMProvider


class TestNormalizeSampleLabel:
    def test_maps_integer_labels(self) -> None:
        assert normalize_sample_label(1) == "FAKE"
        assert normalize_sample_label(0) == "REAL"

    def test_maps_string_labels(self) -> None:
        assert normalize_sample_label("fake") == "FAKE"
        assert normalize_sample_label("real") == "REAL"
        assert normalize_sample_label("TRUE") == "REAL"
        assert normalize_sample_label("FALSE") == "FAKE"

    def test_returns_none_for_unknown_label(self) -> None:
        assert normalize_sample_label("mixed") is None


class TestExtractViFactCheckSample:
    def test_extracts_huggingface_schema(self) -> None:
        sample = extract_vifactcheck_sample(
            {
                "Statement": "Việt Nam đạt tăng trưởng GDP 50% trong quý 1 năm 2024.",
                "Evidence": "Số liệu này không xuất hiện trong báo cáo chính thức.",
                "labels": 1,
            },
            sample_id=7,
        )

        assert sample == {
            "claim": "Việt Nam đạt tăng trưởng GDP 50% trong quý 1 năm 2024.",
            "evidence": "Số liệu này không xuất hiện trong báo cáo chính thức.",
            "label": "FAKE",
            "id": 7,
        }

    def test_extracts_legacy_schema(self) -> None:
        sample = extract_vifactcheck_sample(
            {
                "claim": "World Bank báo cáo GDP Việt Nam tăng 5.6% năm 2023.",
                "evidence": "Báo cáo World Bank 2023.",
                "label": "REAL",
            },
            sample_id=3,
        )

        assert sample == {
            "claim": "World Bank báo cáo GDP Việt Nam tăng 5.6% năm 2023.",
            "evidence": "Báo cáo World Bank 2023.",
            "label": "REAL",
            "id": 3,
        }

    def test_skips_sample_without_claim_text(self) -> None:
        sample = extract_vifactcheck_sample(
            {"Statement": "", "Evidence": "evidence", "labels": 1},
            sample_id=1,
        )

        assert sample is None


class TestBenchmarkParsingHelpers:
    def test_parse_verdict_label(self) -> None:
        assert parse_verdict_label("FAKE") == "FAKE"
        assert parse_verdict_label("This looks real") == "REAL"
        assert parse_verdict_label("uncertain") == "UNCERTAIN"
        assert parse_verdict_label("n/a") == "UNKNOWN"

    def test_extract_multi_agent_verdict_prefers_json(self) -> None:
        text = '{"verdict": "fake", "confidence": 0.9}'
        assert extract_multi_agent_verdict(text) == "FAKE"

    def test_score_prediction_maps_unverifiable_to_real(self) -> None:
        assert _score_prediction("UNCERTAIN", "REAL") == (False, True)
        assert _score_prediction("UNCERTAIN", "FAKE") == (False, True)
        assert _score_prediction("REAL", "REAL") == (True, False)


class TestCreateBenchmarkModel:
    def test_uses_shared_factory_for_google_provider(self, monkeypatch) -> None:
        factory = Mock(return_value="google-model")
        monkeypatch.setattr("scripts.benchmark.create_chat_model", factory)
        monkeypatch.setattr(
            "scripts.benchmark.get_llm_config",
            lambda: LLMConfig(
                provider=LLMProvider.GEMINI_GOOGLE,
                model="gemini-2.0-flash",
            ),
        )

        result = create_benchmark_model(timeout_seconds=15)

        assert result == "google-model"
        factory.assert_called_once()

    def test_uses_openai_client_for_nvidia_provider(self, monkeypatch) -> None:
        openai_ctor = Mock(return_value="nvidia-client")
        monkeypatch.setattr("openai.OpenAI", openai_ctor)
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
        monkeypatch.setattr(
            "scripts.benchmark.get_llm_config",
            lambda: LLMConfig(
                provider=LLMProvider.GEMINI_NVIDIA,
                model="google/gemma-3n-e4b-it",
            ),
        )

        result = create_benchmark_model(timeout_seconds=20)

        assert result == "nvidia-client"
        openai_ctor.assert_called_once_with(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="nv-key",
            timeout=20.0,
        )

    def test_uses_openai_client_for_openai_provider(self, monkeypatch) -> None:
        openai_ctor = Mock(return_value="openai-client")
        monkeypatch.setattr("openai.OpenAI", openai_ctor)
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        monkeypatch.setattr(
            "scripts.benchmark.get_llm_config",
            lambda: LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        result = create_benchmark_model(timeout_seconds=12)

        assert result == "openai-client"
        openai_ctor.assert_called_once_with(
            api_key="oa-key",
            base_url="https://openrouter.ai/api/v1",
            timeout=12.0,
        )

    def test_returns_langchain_like_content(self) -> None:
        class LangChainLikeModel:
            def __init__(self) -> None:
                self.invoke = Mock(return_value=SimpleNamespace(content="REAL"))

        llm = LangChainLikeModel()

        from scripts.benchmark import invoke_text_model

        result = invoke_text_model(llm, system_prompt="sys", user_prompt="user")

        assert result == "REAL"
        llm.invoke.assert_called_once()

    def test_returns_openai_message_content(self, monkeypatch) -> None:
        message = SimpleNamespace(content="FAKE", reasoning_content="")
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        llm = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=response))
            )
        )
        monkeypatch.setattr(
            "scripts.benchmark.get_llm_config",
            lambda: LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        from scripts.benchmark import invoke_text_model

        result = invoke_text_model(llm, system_prompt="sys", user_prompt="user")

        assert result == "FAKE"
        llm.chat.completions.create.assert_called_once()

    def test_returns_reasoning_content_for_nvidia(self, monkeypatch) -> None:
        message = SimpleNamespace(content=None, reasoning_content="REAL")
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        llm = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=response))
            )
        )
        monkeypatch.setattr(
            "scripts.benchmark.get_llm_config",
            lambda: LLMConfig(
                provider=LLMProvider.GEMINI_NVIDIA,
                model="google/gemma-3n-e4b-it",
            ),
        )

        from scripts.benchmark import invoke_text_model

        result = invoke_text_model(llm, system_prompt="sys", user_prompt="user")

        assert result == "REAL"
        llm.chat.completions.create.assert_called_once()

    def test_raises_clear_error_when_nvidia_key_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.setattr(
            "scripts.benchmark.get_llm_config",
            lambda: LLMConfig(
                provider=LLMProvider.GEMINI_NVIDIA,
                model="google/gemma-3n-e4b-it",
            ),
        )

        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            create_benchmark_model(timeout_seconds=10)

    def test_raises_clear_error_when_openai_key_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            "scripts.benchmark.get_llm_config",
            lambda: LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_benchmark_model(timeout_seconds=10)

    def test_google_provider_does_not_require_timeout_specific_openai_client(
        self,
        monkeypatch,
    ) -> None:
        factory = Mock(return_value="google-model")
        monkeypatch.setattr("scripts.benchmark.create_chat_model", factory)
        monkeypatch.setattr(
            "scripts.benchmark.get_llm_config",
            lambda: LLMConfig(
                provider=LLMProvider.GEMINI_GOOGLE,
                model="gemini-2.0-flash",
            ),
        )

        result = create_benchmark_model(timeout_seconds=30)

        assert result == "google-model"
        factory.assert_called_once()

    def test_invoke_text_model_falls_back_to_string_representation(self) -> None:
        class LangChainLikeModel:
            def __init__(self) -> None:
                self.invoke = Mock(return_value=123)

        llm = LangChainLikeModel()

        from scripts.benchmark import invoke_text_model

        result = invoke_text_model(llm, system_prompt="sys", user_prompt="user")

        assert result == "123"
