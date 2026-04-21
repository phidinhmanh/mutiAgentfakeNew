"""Tests for PhoBERT baseline model."""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from fake_news_detector.models.baseline import LABEL_MAP, PhoBERTBaseline


class TestLabelMap:
    """Test LABEL_MAP constant."""

    def test_label_map_has_real(self) -> None:
        """LABEL_MAP contains REAL."""
        assert 0 in LABEL_MAP
        assert LABEL_MAP[0] == "REAL"

    def test_label_map_has_fake(self) -> None:
        """LABEL_MAP contains FAKE."""
        assert 1 in LABEL_MAP
        assert LABEL_MAP[1] == "FAKE"


class TestPhoBERTBaselineInit:
    """Test PhoBERTBaseline initialization."""

    @patch("fake_news_detector.models.baseline.AutoModelForSequenceClassification")
    @patch("fake_news_detector.models.baseline.AutoTokenizer")
    @patch("fake_news_detector.models.baseline.settings")
    def test_init_loads_tokenizer(
        self, mock_settings: Mock, mock_tokenizer: Mock, mock_model: Mock
    ) -> None:
        """Initializes tokenizer."""
        mock_settings.phobert_model = "vinai/phobert-base"
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        model = PhoBERTBaseline()

        mock_tokenizer.from_pretrained.assert_called_once_with("vinai/phobert-base")
        mock_model.from_pretrained.assert_called_once()

    @patch("fake_news_detector.models.baseline.AutoModelForSequenceClassification")
    @patch("fake_news_detector.models.baseline.AutoTokenizer")
    @patch("fake_news_detector.models.baseline.settings")
    def test_init_loads_model(
        self, mock_settings: Mock, mock_tokenizer: Mock, mock_model: Mock
    ) -> None:
        """Initializes model with correct config."""
        mock_settings.phobert_model = "vinai/phobert-base"
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        model = PhoBERTBaseline()

        call_kwargs = mock_model.from_pretrained.call_args.kwargs
        assert call_kwargs.get("num_labels") == 2
        assert call_kwargs.get("ignore_mismatched_sizes") is True

    @patch("fake_news_detector.models.baseline.AutoModelForSequenceClassification")
    @patch("fake_news_detector.models.baseline.AutoTokenizer")
    @patch("fake_news_detector.models.baseline.settings")
    def test_init_sets_model_eval(
        self, mock_settings: Mock, mock_tokenizer: Mock, mock_model: Mock
    ) -> None:
        """Sets model to eval mode."""
        mock_settings.phobert_model = "vinai/phobert-base"
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        model = PhoBERTBaseline()
        mock_model_instance.eval.assert_called_once()


class TestPhoBERTBaselineEmptyInput:
    """Test PhoBERTBaseline with empty input."""

    @patch("fake_news_detector.models.baseline.AutoModelForSequenceClassification")
    @patch("fake_news_detector.models.baseline.AutoTokenizer")
    @patch("fake_news_detector.models.baseline.settings")
    def test_predict_empty_text(
        self, mock_settings: Mock, mock_tokenizer: Mock, mock_model: Mock
    ) -> None:
        """Empty text returns UNKNOWN label."""
        mock_settings.phobert_model = "vinai/phobert-base"
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = PhoBERTBaseline()
        result = model.predict("")
        assert result["label"] == "UNKNOWN"
        assert result["confidence"] == 0.0
        assert result["fake_prob"] == 0.5
        assert result["real_prob"] == 0.5

    @patch("fake_news_detector.models.baseline.AutoModelForSequenceClassification")
    @patch("fake_news_detector.models.baseline.AutoTokenizer")
    @patch("fake_news_detector.models.baseline.settings")
    def test_predict_whitespace_only(
        self, mock_settings: Mock, mock_tokenizer: Mock, mock_model: Mock
    ) -> None:
        """Whitespace-only returns UNKNOWN."""
        mock_settings.phobert_model = "vinai/phobert-base"
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = PhoBERTBaseline()
        result = model.predict("   \n\t   ")
        assert result["label"] == "UNKNOWN"

    @patch("fake_news_detector.models.baseline.AutoModelForSequenceClassification")
    @patch("fake_news_detector.models.baseline.AutoTokenizer")
    @patch("fake_news_detector.models.baseline.settings")
    def test_predict_with_sliding_window_empty(
        self, mock_settings: Mock, mock_tokenizer: Mock, mock_model: Mock
    ) -> None:
        """Sliding window with empty text returns UNKNOWN."""
        mock_settings.phobert_model = "vinai/phobert-base"
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = PhoBERTBaseline()
        result = model.predict_with_sliding_window("")
        assert result["label"] == "UNKNOWN"


class TestGetBaselineModel:
    """Test get_baseline_model function."""

    def test_returns_baseline_instance(self) -> None:
        """Returns PhoBERTBaseline instance."""
        from fake_news_detector.models.baseline import get_baseline_model

        with patch(
            "fake_news_detector.models.baseline._baseline_model", None
        ):
            with patch(
                "fake_news_detector.models.baseline.PhoBERTBaseline.__init__",
                return_value=None,
            ):
                model = get_baseline_model()
                assert isinstance(model, PhoBERTBaseline)

    def test_singleton_pattern(self) -> None:
        """Returns same instance on multiple calls."""
        from fake_news_detector.models.baseline import get_baseline_model

        mock_model = Mock(spec=PhoBERTBaseline)
        with patch(
            "fake_news_detector.models.baseline._baseline_model", mock_model
        ):
            result = get_baseline_model()
            assert result is mock_model