"""Word cloud visualization for text analysis."""
import logging
from typing import Any

from wordcloud import WordCloud

from fake_news_detector.data.preprocessing import tokenize_words

logger = logging.getLogger(__name__)


def generate_wordcloud(
    text: str,
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
) -> WordCloud:
    """Generate word cloud from text.

    Args:
        text: Input text
        width: Image width
        height: Image height
        background_color: Background color

    Returns:
        WordCloud object
    """
    if not text:
        return WordCloud(width=width, height=height)

    tokens = tokenize_words(text)
    text_content = " ".join(tokens)

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=100,
        colormap="viridis",
        min_font_size=10,
    )

    return wc.generate(text_content)


def get_top_words(text: str, n: int = 20) -> list[tuple[str, int]]:
    """Get top N words by frequency.

    Args:
        text: Input text
        n: Number of top words

    Returns:
        List of (word, frequency) tuples
    """
    tokens = tokenize_words(text)

    stopwords = {
        "và", "của", "là", "có", "được", "trong", "cho", "với",
        "theo", "này", "đã", "không", "tại", "về", "sau",
        "các", "những", "một", "cũng", "như", "đến",
    }

    filtered = [t for t in tokens if t.lower() not in stopwords and len(t) > 2]

    freq: dict[str, int] = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1

    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_freq[:n]


def analyze_text_length(text: str) -> dict[str, Any]:
    """Analyze text length statistics.

    Args:
        text: Input text

    Returns:
        Dictionary with length statistics
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
        }

    tokens = tokenize_words(text)
    words = text.split()

    from fake_news_detector.data.preprocessing import split_sentences
    sentences = split_sentences(text)

    char_count = len(text)
    word_count = len(tokens)
    sentence_count = len(sentences)
    avg_word_length = sum(len(w) for w in tokens) / word_count if word_count > 0 else 0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
    }