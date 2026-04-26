"""Tests for data loader."""

from __future__ import annotations

from unittest.mock import Mock, patch

from fake_news_detector.data.loader import (
    DATASET_NAME,
    format_sample,
    get_sample,
    load_vifactcheck,
)


class TestLoadViFactCheck:
    """Test load_vifactcheck function."""

    @patch("fake_news_detector.data.loader.load_dataset")
    def test_load_train_split(self, mock_load: Mock) -> None:
        """Loads train split by default."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load.return_value = mock_dataset

        load_vifactcheck("train")
        mock_load.assert_called_once()
        assert mock_load.call_args[1].get("split") == "train" or "train" in str(
            mock_load.call_args
        )

    @patch("fake_news_detector.data.loader.load_dataset")
    def test_load_dataset_name(self, mock_load: Mock) -> None:
        """Loads correct dataset name."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load.return_value = mock_dataset

        load_vifactcheck("train")
        call_args = mock_load.call_args
        assert DATASET_NAME in str(call_args) or call_args[0][0] == DATASET_NAME

    @patch("fake_news_detector.data.loader.load_dataset")
    def test_returns_dataset_object(self, mock_load: Mock) -> None:
        """Returns dataset object."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load.return_value = mock_dataset

        result = load_vifactcheck()
        assert result is mock_dataset


class TestGetSample:
    """Test get_sample function."""

    @patch("fake_news_detector.data.loader.load_vifactcheck")
    def test_get_sample_returns_dict(self, mock_load: Mock) -> None:
        """Returns dictionary with expected keys."""
        mock_sample = {
            "claim": "Test claim",
            "evidence": "Test evidence",
            "label": "REAL",
            "claim_date": "2024-01-01",
            "source": "test_source",
        }
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value=mock_sample)
        mock_load.return_value = mock_dataset

        result = get_sample(0)
        assert isinstance(result, dict)

    @patch("fake_news_detector.data.loader.load_vifactcheck")
    def test_get_sample_has_required_fields(self, mock_load: Mock) -> None:
        """Sample has claim, evidence, label fields."""
        mock_sample = {
            "claim": "Test claim",
            "evidence": "Test evidence",
            "label": "FAKE",
        }
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value=mock_sample)
        mock_load.return_value = mock_dataset

        result = get_sample(0)
        assert "claim" in result
        assert "evidence" in result
        assert "label" in result


class TestFormatSample:
    """Test format_sample function."""

    def test_format_complete_sample(self) -> None:
        """Formats complete sample correctly."""
        sample = {
            "claim": "Vietnam GDP grew 8%",
            "evidence": "World Bank report",
            "label": "REAL",
            "source": "vnexpress",
            "claim_date": "2024-01-15",
        }
        result = format_sample(sample)
        assert "Vietnam GDP grew 8%" in result
        assert "World Bank report" in result
        assert "REAL" in result
        assert "vnexpress" in result

    def test_format_minimal_sample(self) -> None:
        """Formats minimal sample with defaults."""
        sample = {
            "claim": "Test",
            "evidence": "Evidence",
            "label": "FAKE",
        }
        result = format_sample(sample)
        assert "Test" in result
        assert "N/A" in result
