"""Shared NLP utilities for claim extraction tools."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("TRUST_agents.agents.claim_extractor_tools")

VIETNAMESE_DIACRITIC_MARKERS = [
    "ă",
    "â",
    "đ",
    "ê",
    "ô",
    "ơ",
    "ư",
    "ạ",
    "ả",
    "ấ",
    "ầ",
    "ẩ",
    "ẫ",
    "ậ",
    "ắ",
    "ằ",
    "ẳ",
    "ẵ",
    "ặ",
    "ẹ",
    "ẻ",
    "ẽ",
    "ế",
    "ề",
    "ể",
    "ễ",
    "ệ",
    "ỉ",
    "ị",
    "ọ",
    "ỏ",
    "ố",
    "ồ",
    "ổ",
    "ỗ",
    "ộ",
    "ụ",
    "ủ",
    "ứ",
    "ừ",
    "ử",
    "ữ",
    "ự",
    "ợ",
    "tôi",
    "bạn",
    "ông",
    "bà",
    "chúng",
    "họ",
    "năm",
    "tháng",
    "ngày",
    "giờ",
    "phút",
]

VIETNAMESE_FACTUAL_MARKERS = [
    "là",
    "có",
    "đã",
    "sẽ",
    "đang",
    "phát",
    "triển",
    "tăng",
    "giảm",
    "cho biết",
    "theo",
    "báo",
    "tin",
    "nói",
    "khẳng",
    "định",
    "công",
    "bố",
    "thông",
    "tin",
    "nghiên",
    "cứu",
    "xác",
    "nhận",
]

VIETNAMESE_CLAIM_MARKERS = [
    "nói",
    "cho biết",
    "khẳng định",
    "tuyên bố",
    "thông báo",
    "báo cáo",
    "xác nhận",
    "phủ nhận",
    "công bố",
    "kết luận",
    "nghiên cứu",
    "cho hay",
    "trả lời",
    "hồi",
    "tin",
    "theo",
    "được",
    "là",
    "có",
    "đã",
    "sẽ",
    "đang",
    "phát",
    "triển",
    "tăng",
    "giảm",
    "đạt",
    "vượt",
    "hạ",
]

ENGLISH_CLAIM_VERBS = {
    "say",
    "claim",
    "state",
    "report",
    "announce",
    "declare",
    "assert",
    "allege",
    "argue",
    "maintain",
    "contend",
    "insist",
    "affirm",
    "attest",
    "testify",
    "reveal",
    "disclose",
    "admit",
    "acknowledge",
    "confess",
    "confirm",
    "deny",
    "show",
    "prove",
    "demonstrate",
    "indicate",
    "suggest",
    "imply",
    "note",
    "observe",
    "find",
    "discover",
    "detect",
    "notice",
    "emphasize",
    "stress",
    "highlight",
    "underscore",
    "point out",
    "predict",
    "forecast",
    "warn",
    "caution",
    "anticipate",
    "expect",
    "explain",
    "describe",
    "characterize",
    "define",
    "specify",
    "believe",
    "think",
    "consider",
    "regard",
    "view",
    "estimate",
    "calculate",
    "determine",
    "assess",
    "evaluate",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "contain",
    "include",
    "involve",
}

VI_FALLBACK_CLAIM_MARKERS = [" là ", " có ", " đã ", " sẽ ", " đang ", " theo "]


def detect_language(text: str) -> str:
    lower_text = text.lower()
    vietnamese_count = sum(1 for marker in VIETNAMESE_DIACRITIC_MARKERS if marker in lower_text)
    has_diacritics = any(c in lower_text for c in "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộụủứừửữựợ")
    if vietnamese_count >= 2 or has_diacritics:
        return "vi"
    return "en"


def load_spacy_model(*model_names: str):
    try:
        import spacy
    except ImportError:
        logger.warning("spaCy not installed, using heuristic fallback")
        return None

    for model_name in model_names:
        try:
            return spacy.load(model_name)
        except OSError:
            logger.warning("spaCy model not found: %s", model_name)

    logger.warning("No spaCy model available, using heuristic fallback")
    return None


def sentencize_vietnamese(text: str) -> list[str]:
    try:
        from underthesea import sent_tokenize

        return sent_tokenize(text)
    except ImportError:
        logger.warning("underthesea not installed, using regex-based splitting")
        sentences = re.split(r'[。.!?"]|\n+', text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        logger.warning("underthesea sent_tokenize failed: %s", e)
        return [text]


def tokenize_vietnamese(text: str) -> list[str]:
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text").split()
    except ImportError:
        logger.warning("underthesea not installed, using simple split")
        return text.split()
    except Exception as e:
        logger.warning("underthesea tokenization failed: %s", e)
        return text.split()


def fallback_claim_sentences(text: str) -> list[str]:
    lang = detect_language(text)
    if lang == "vi":
        sentences = sentencize_vietnamese(text)
    else:
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]

    claims: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped or stripped.endswith(("?", "!")):
            continue
        if len(stripped.split()) < 4:
            continue
        if lang == "vi":
            lowered = stripped.lower()
            if any(marker in lowered for marker in VI_FALLBACK_CLAIM_MARKERS):
                claims.append(stripped)
        elif re.search(
            r"\b(is|are|was|were|has|have|had|said|says|reported|announced|according to)\b",
            stripped,
            re.IGNORECASE,
        ):
            claims.append(stripped)
    return claims


def looks_like_claim_vietnamese(doc) -> bool:
    text = doc.text.strip()
    if not text or text.endswith("?") or text.endswith("!"):
        return False
    if len(doc) < 3:
        return False
    text_lower = text.lower()
    return any(marker in text_lower for marker in VIETNAMESE_FACTUAL_MARKERS)


def looks_like_claim_english(doc) -> bool:
    text = doc.text.strip()
    if text.endswith("?") or text.endswith("!"):
        return False
    if len(doc) < 3:
        return False
    return any(token.pos_ == "VERB" for token in doc)
