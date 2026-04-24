from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fake_news_detector import app
from fake_news_detector.application import analysis_service


class TestAnalyzeArticle:
    """Characterization tests for current app analysis behavior."""

    def test_trust_pipeline_returns_expected_shape(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "TRUST_AVAILABLE", True)
        monkeypatch.setattr(app, "LEGACY_AVAILABLE", True)

        baseline_model = Mock()
        baseline_model.predict_with_sliding_window.return_value = {
            "label": "REAL",
            "confidence": 0.8,
            "fake_prob": 0.2,
            "real_prob": 0.8,
        }
        monkeypatch.setattr(
            analysis_service,
            "get_baseline_model",
            lambda: baseline_model,
        )
        monkeypatch.setattr(
            analysis_service,
            "extract_stylistic_features",
            lambda text: {"fake_score": 0.1, "num_words": len(text.split())},
        )
        monkeypatch.setattr(
            analysis_service,
            "summarize_for_long_text",
            lambda text, max_chars: text,
        )

        orchestrator = Mock()
        orchestrator.process_text.return_value = SimpleNamespace(
            claims=["claim A"],
            results=[
                {
                    "claim": "claim A",
                    "verdict": "true",
                    "confidence": 0.9,
                    "label": "true",
                    "reasoning": "supported",
                }
            ],
            summary={"total_claims": 1, "verdicts": {"true": 1}},
        )
        monkeypatch.setattr(
            app,
            "TRUSTOrchestrator",
            lambda top_k_evidence=5: orchestrator,
        )

        result = app.analyze_article("Bài báo cần kiểm tra", use_trust=True)

        assert "baseline" in result
        assert "trust" in result
        assert result["claims"] == [{"text": "claim A", "source": "trust"}]
        assert result["verdicts"][0]["verdict"] == "true"
        assert "stylistic_features" in result

    def test_trust_failure_sets_error_and_keeps_baseline(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "TRUST_AVAILABLE", True)
        monkeypatch.setattr(app, "LEGACY_AVAILABLE", False)

        baseline_model = Mock()
        baseline_model.predict_with_sliding_window.return_value = {"confidence": 0.8}
        monkeypatch.setattr(
            analysis_service,
            "get_baseline_model",
            lambda: baseline_model,
        )
        monkeypatch.setattr(
            analysis_service,
            "extract_stylistic_features",
            lambda text: {"fake_score": 0.1},
        )
        monkeypatch.setattr(
            analysis_service,
            "summarize_for_long_text",
            lambda text, max_chars: text,
        )

        orchestrator = Mock()
        orchestrator.process_text.side_effect = RuntimeError("trust failed")
        monkeypatch.setattr(
            app,
            "TRUSTOrchestrator",
            lambda top_k_evidence=5: orchestrator,
        )

        result = app.analyze_article("Bài báo cần kiểm tra", use_trust=True)

        assert "baseline" in result
        assert result["trust_error"] == "trust failed"
        assert result["claims"] == []

    def test_invalid_trust_orchestrator_raises_clear_error(self, monkeypatch) -> None:
        baseline_model = Mock()
        baseline_model.predict_with_sliding_window.return_value = {"confidence": 0.8}
        monkeypatch.setattr(
            analysis_service,
            "get_baseline_model",
            lambda: baseline_model,
        )
        monkeypatch.setattr(
            analysis_service,
            "extract_stylistic_features",
            lambda text: {"fake_score": 0.1},
        )
        monkeypatch.setattr(
            analysis_service,
            "summarize_for_long_text",
            lambda text, max_chars: text,
        )

        class BrokenOrchestrator:
            def process_text(self, article: str):
                raise NotImplementedError

        with pytest.raises(TypeError, match="Invalid TRUST orchestrator"):
            analysis_service.analyze_with_trust(
                article="Bài báo cần kiểm tra",
                orchestrator=BrokenOrchestrator(),
            )

    def test_legacy_pipeline_enriches_and_flattens_evidence(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "TRUST_AVAILABLE", False)
        monkeypatch.setattr(app, "LEGACY_AVAILABLE", True)

        baseline_model = Mock()
        baseline_model.predict_with_sliding_window.return_value = {"confidence": 0.8}
        monkeypatch.setattr(
            analysis_service,
            "get_baseline_model",
            lambda: baseline_model,
        )
        monkeypatch.setattr(
            analysis_service,
            "extract_stylistic_features",
            lambda text: {"fake_score": 0.1},
        )
        monkeypatch.setattr(
            analysis_service,
            "summarize_for_long_text",
            lambda text, max_chars: text,
        )

        claims = [{"text": "claim A", "type": "FACT", "verifiable": True}]
        monkeypatch.setattr(app, "extract_claims", lambda article: claims)
        monkeypatch.setattr(
            app,
            "filter_verifiable_claims",
            lambda extracted: extracted,
        )
        monkeypatch.setattr(
            app,
            "retrieve_evidence_for_claims",
            lambda verifiable: [{"text": "claim A", "evidence": [{"content": "ev1"}]}],
        )
        monkeypatch.setattr(
            app,
            "enrich_evidence_with_context",
            lambda evidence, text: [{"content": "ev1", "source": text}],
        )

        result = app.analyze_article("Bài báo cần kiểm tra", use_trust=False)

        assert result["claims"] == claims
        assert result["verifiable_claims"] == claims
        assert result["claims_with_evidence"][0]["evidence"][0]["source"] == "claim A"
        assert result["evidence"][0]["content"] == "ev1"

    def test_long_article_is_summarized_before_analysis(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "TRUST_AVAILABLE", False)
        monkeypatch.setattr(app, "LEGACY_AVAILABLE", True)
        monkeypatch.setattr(
            analysis_service,
            "extract_stylistic_features",
            lambda text: {"fake_score": 0.1},
        )

        summarize_mock = Mock(return_value="shortened")
        monkeypatch.setattr(
            analysis_service,
            "summarize_for_long_text",
            summarize_mock,
        )

        baseline_model = Mock()
        baseline_model.predict_with_sliding_window.return_value = {"confidence": 0.8}
        monkeypatch.setattr(
            analysis_service,
            "get_baseline_model",
            lambda: baseline_model,
        )
        monkeypatch.setattr(app, "extract_claims", lambda article: [])
        monkeypatch.setattr(app, "filter_verifiable_claims", lambda extracted: [])

        article = "x" * 4000
        result = app.analyze_article(article, use_trust=False)

        summarize_mock.assert_called_once_with(article, max_chars=3000)
        baseline_model.predict_with_sliding_window.assert_called_once_with("shortened")
        assert result["summarized"] is True
