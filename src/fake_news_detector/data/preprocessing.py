"""Text preprocessing for Vietnamese text using underthesea."""

import re

from underthesea import sent_tokenize, word_tokenize


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def split_sentences(text: str) -> list[str]:
    """Split Vietnamese text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    if not text:
        return []

    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str) -> list[str]:
    """Tokenize Vietnamese text into words.

    Args:
        text: Input text

    Returns:
        List of words
    """
    if not text:
        return []

    tokens = word_tokenize(text, format="text").split()
    return [t.strip() for t in tokens if t.strip()]


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named entities from Vietnamese text.

    Args:
        text: Input text

    Returns:
        Dictionary with entity types and their values
    """
    from underthesea import pos_tag

    if not text:
        return {"PER": [], "LOC": [], "ORG": [], "NUM": []}

    pos_tags = pos_tag(text)
    entities: dict[str, list[str]] = {
        "PER": [],
        "LOC": [],
        "ORG": [],
        "NUM": [],
    }

    for word, tag in pos_tags:
        if tag == "Np":  # Proper noun
            entities["PER"].append(word)
        elif tag == "N":  # Noun - could be location/org
            if _is_location(word):
                entities["LOC"].append(word)
            else:
                entities["ORG"].append(word)
        elif tag == "M":  # Number
            entities["NUM"].append(word)

    return entities


def _is_location(word: str) -> bool:
    """Check if a word is likely a location name."""
    location_markers = {
        "tỉnh",
        "thành phố",
        "tp",
        "huyện",
        "quận",
        "xã",
        "phường",
        "miền",
        "vùng",
        "đông",
        "tây",
        "nam",
        "bắc",
    }
    return any(marker in word.lower() for marker in location_markers)


def extract_numbers(text: str) -> list[str]:
    """Extract all numbers (including percentages, dates) from text.

    Args:
        text: Input text

    Returns:
        List of number strings found
    """
    if not text:
        return []

    patterns = [
        r"\d+\.?\d*%",  # Percentages
        r"\d{1,3}(?:,\d{3})*(?:\.\d+)?",  # Large numbers
        r"\d{1,2}/\d{1,2}/\d{2,4}",  # Dates
        r"\d+(?:\.\d+)?",  # Simple decimals
    ]

    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        numbers.extend(matches)

    return numbers


def summarize_for_long_text(text: str, max_chars: int = 2000) -> str:
    """Summarize long text to fit within token limits.

    Args:
        text: Input text
        max_chars: Maximum characters to keep

    Returns:
        Summarized text preserving key entities and numbers
    """
    if len(text) <= max_chars:
        return text

    sentences = split_sentences(text)
    summary_parts = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_chars:
            summary_parts.append(sentence)
            current_length += sentence_length
        else:
            remaining = max_chars - current_length
            if remaining > 100:
                summary_parts.append(sentence[:remaining] + "...")
            break

    return " ".join(summary_parts)


def preprocess_for_embedding(text: str) -> str:
    """Preprocess text for embedding model.

    Args:
        text: Input text

    Returns:
        Preprocessed text
    """
    text = clean_text(text)
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunk_windows(
    text: str, chunk_size: int = 200, overlap: int = 50
) -> list[str]:
    """Create overlapping chunks for long text processing.

    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if start > 0:
            chunk = "... " + chunk
        if end < len(text):
            chunk = chunk + " ..."

        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
