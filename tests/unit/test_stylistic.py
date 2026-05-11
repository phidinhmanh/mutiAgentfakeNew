"""Tests for stylistic feature extraction."""

from fake_news_detector.models.stylistic import extract_stylistic_features


class TestStylisticFeatures:
    """Test stylistic feature extraction."""

    def test_extract_features_normal_text(self):
        """Test extraction from normal article."""
        text = """
        Theo báo cáo mới nhất, kinh tế Việt Nam tăng trưởng 5.6% trong quý đầu năm.
        Chính phủ đã triển khai nhiều chính sách hỗ trợ doanh nghiệp.
        """
        features = extract_stylistic_features(text)
        assert "num_sentences" in features
        assert "caps_ratio" in features
        assert "fake_score" in features
        assert 0 <= features["fake_score"] <= 1

    def test_extract_features_empty_text(self):
        """Test extraction from empty text."""
        features = extract_stylistic_features("")
        assert features["num_sentences"] == 0
        assert features["fake_score"] == 0.5

    def test_extract_features_emotional_text(self):
        """Test detection of emotional markers."""
        text = "SHOCKING! This is absolutely incredible! You won't believe this!"
        features = extract_stylistic_features(text)
        assert features["caps_ratio"] > 0
        assert features["emotional_markers"] > 0

    def test_extract_features_source_mentions(self):
        """Test source mention detection."""
        text = "Theo Reuters, ngày hôm qua xảy ra sự kiện quan trọng."
        features = extract_stylistic_features(text)
        assert features["source_mentions"] >= 1
