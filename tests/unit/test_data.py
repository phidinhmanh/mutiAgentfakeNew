"""Tests for data module."""
import pytest

from fake_news_detector.data.preprocessing import (
    clean_text,
    split_sentences,
    extract_numbers,
    extract_entities,
    tokenize_words,
    create_chunk_windows,
    summarize_for_long_text,
)


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_clean_text_removes_extra_whitespace(self):
        """Test whitespace normalization."""
        assert clean_text("  Hello   world  ") == "Hello world"
        assert clean_text("") == ""

    def test_split_sentences_vietnamese(self):
        """Test Vietnamese sentence splitting."""
        text = "Đây là câu thứ nhất. Đây là câu thứ hai!"
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_tokenize_words(self):
        """Test word tokenization."""
        text = "Xin chào thế giới"
        tokens = tokenize_words(text)
        assert len(tokens) >= 2

    def test_extract_numbers(self):
        """Test number extraction."""
        text = "Tỷ lệ tăng 25% và đạt 1.5 triệu đồng"
        numbers = extract_numbers(text)
        assert len(numbers) >= 2
        assert "25%" in numbers or "25" in numbers

    def test_create_chunk_windows_basic(self):
        """Test sliding window chunking."""
        text = "a" * 500
        chunks = create_chunk_windows(text, chunk_size=200, overlap=50)
        assert len(chunks) >= 2

    def test_create_chunk_windows_short_text(self):
        """Test chunking short text returns single chunk."""
        text = "short"
        chunks = create_chunk_windows(text, chunk_size=200)
        assert len(chunks) == 1

    def test_summarize_for_long_text_short(self):
        """Test summarization of short text."""
        text = "This is short text."
        result = summarize_for_long_text(text, max_chars=100)
        assert result == text

    def test_summarize_for_long_text_truncates(self):
        """Test summarization truncation."""
        text = "This is a very long text. " * 50
        result = summarize_for_long_text(text, max_chars=100)
        assert len(result) <= 120