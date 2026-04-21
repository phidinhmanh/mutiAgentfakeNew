# -*- coding: utf-8 -*-

"""
Claim Extractor Tools - Tools for NLP-based claim extraction.

Tools used by the Claim Extractor ReAct Agent:
- NER Tool: Named Entity Recognition
- Dependency Parsing Tool: Sentence structure analysis
- LLM Tool: Zero-shot reasoning

Each tool is self-contained, logs its actions, and returns status summaries.
Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import os
import re
import json
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from trust_agents.llm.llm_helpers import call_llm, call_llm_json

load_dotenv()
logger = logging.getLogger("TRUST_agents.agents.claim_extractor_tools")
logger.propagate = True


def _detect_language(text: str) -> str:
    """Detect if text is Vietnamese or English.

    Args:
        text: Input text

    Returns:
        'vi' for Vietnamese, 'en' for English
    """
    # Vietnamese common patterns
    vietnamese_markers = [
        "ă", "â", "đ", "ê", "ô", "ơ", "ư", "ạ", "ả", "ấ", "ầ", "ẩ", "ẫ",
        "ậ", "ắ", "ằ", "ẳ", "ẵ", "ặ", "ẹ", "ẻ", "ẽ", "ế", "ề", "ể", "ễ",
        "ệ", "ỉ", "ị", "ọ", "ỏ", "ố", "ồ", "ổ", "ỗ", "ộ", "ụ", "ủ", "ứ",
        "ừ", "ử", "ữ", "ự", "ợ", "tôi", "bạn", "ông", "bà", "chúng", "họ",
        "năm", "tháng", "ngày", "giờ", "phút"
    ]

    lower_text = text.lower()
    vietnamese_count = sum(1 for marker in vietnamese_markers if marker in lower_text)

    # Also check for Vietnamese diacritics
    has_diacritics = any(c in lower_text for c in "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộụủứừửữựợ")

    if vietnamese_count >= 2 or has_diacritics:
        return "vi"
    return "en"


def _tokenize_vietnamese(text: str) -> list[str]:
    """Tokenize Vietnamese text using underthesea.

    Args:
        text: Vietnamese text to tokenize

    Returns:
        List of tokens
    """
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").split()
    except ImportError:
        logger.warning("underthesea not installed, using simple split")
        return text.split()
    except Exception as e:
        logger.warning(f"underthesea tokenization failed: {e}")
        return text.split()


def _sentencize_vietnamese(text: str) -> list[str]:
    """Split Vietnamese text into sentences.

    Args:
        text: Vietnamese text

    Returns:
        List of sentences
    """
    try:
        from underthesea import sent_tokenize
        return sent_tokenize(text)
    except ImportError:
        logger.warning("underthesea not installed, using regex-based splitting")
        # Fallback: split by common Vietnamese punctuation
        sentences = re.split(r'[。.!?"]|{1,2}', text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        logger.warning(f"underthesea sent_tokenize failed: {e}")
        return [text]


@tool()
async def ner_claim_extraction_tool(text: str) -> str:
    """
    Extract claims from text using Named Entity Recognition (NER).
    Identifies sentences containing named entities (people, organizations, locations, etc.).
    Supports both English (spaCy) and Vietnamese (underthesea).

    Args:
        text: Input text to extract claims from

    Returns:
        JSON string with claims list and method
    """
    logger.info(f"[DEBUG] ner_claim_extraction_tool called")

    lang = _detect_language(text)
    logger.info(f"Detected language: {lang}")

    try:
        if lang == "vi":
            return _ner_extract_vietnamese(text)
        return _ner_extract_english(text)
    except Exception as e:
        logger.error(f"Error during NER extraction: {e}")
        result = {"claims": [], "error": str(e)}
        return json.dumps(result)


def _ner_extract_vietnamese(text: str) -> str:
    """Extract claims from Vietnamese text using NER."""
    import spacy
    try:
        nlp = spacy.load("xx_ent_wiki_sm")
    except OSError:
        logger.warning("xx_ent_wiki_sm model not found, using default")
        nlp = spacy.load("xx_sm")

    doc = nlp(text)

    entity_types = ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "EVENT"]
    claims = []

    for sent in _sentencize_vietnamese(text):
        sent_doc = nlp(sent.strip())
        entities = [ent.text for ent in sent_doc.ents if ent.label_ in entity_types]

        if entities and _looks_like_claim_vietnamese(sent_doc):
            claims.append(sent.strip())

    logger.info(f"ner_claim_extraction_tool completed: {len(claims)} claims found")
    result = {"claims": claims, "method": "ner", "language": "vi", "ner_done": True}
    return json.dumps(result, ensure_ascii=False)


def _ner_extract_english(text: str) -> str:
    """Extract claims from English text using NER."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    entity_types = ["PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "MONEY"]
    claims = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_doc = nlp(sent_text)

        entities = [ent.text for ent in sent_doc.ents if ent.label_ in entity_types]

        if entities and _looks_like_claim(sent_doc):
            claims.append(sent_text)

    logger.info(f"ner_claim_extraction_tool completed: {len(claims)} claims found")
    result = {"claims": claims, "method": "ner", "language": "en", "ner_done": True}
    return json.dumps(result)


def _looks_like_claim_vietnamese(doc) -> bool:
    """Check if a Vietnamese sentence looks like a factual claim."""
    text = doc.text.strip()
    if not text:
        return False

    # Skip questions and exclamations
    if text.endswith("?") or text.endswith("!"):
        return False

    # Skip very short sentences
    if len(doc) < 3:
        return False

    # Check for factual markers in Vietnamese
    factual_markers = [
        "là", "có", "đã", "sẽ", "đang", "phát", "triển", "tăng", "giảm",
        "cho biết", "theo", "báo", "tin", "nói", "khẳng", "định", "công",
        "bố", "thông", "tin", "nghiên", "cứu", "xác", "nhận"
    ]

    text_lower = text.lower()
    return any(marker in text_lower for marker in factual_markers)


@tool()
async def dependency_claim_extraction_tool(text: str) -> str:
    """
    Extract claims from text using dependency parsing.
    Identifies claim patterns: subject-verb-object structures with factual content.
    Supports both English (spaCy) and Vietnamese (underthesea).

    Args:
        text: Input text to extract claims from

    Returns:
        JSON string with claims list and method
    """
    logger.info(f"[DEBUG] dependency_claim_extraction_tool called")

    lang = _detect_language(text)
    logger.info(f"Detected language: {lang}")

    try:
        if lang == "vi":
            return _dependency_extract_vietnamese(text)
        return _dependency_extract_english(text)
    except Exception as e:
        logger.error(f"Error during dependency parsing: {e}")
        result = {"claims": [], "error": str(e)}
        return json.dumps(result)


def _dependency_extract_vietnamese(text: str) -> str:
    """Extract claims from Vietnamese text using dependency patterns."""
    import spacy

    # Try multilingual model first
    try:
        nlp = spacy.load("xx_ent_wiki_sm")
    except OSError:
        try:
            nlp = spacy.load("xx_sm")
        except OSError:
            logger.warning("No multilingual model, using English model")
            nlp = spacy.load("en_core_web_sm")

    # Vietnamese claim verbs/markers
    vi_claim_markers = [
        "nói", "cho biết", "khẳng định", "tuyên bố", "thông báo", "báo cáo",
        "xác nhận", "phủ nhận", "công bố", "kết luận", "nghiên cứu", "cho hay",
        "trả lời", "hồi", "tin", "theo", "được", "là", "có", "đã", "sẽ", "đang",
        "phát", "triển", "tăng", "giảm", "đạt", "vượt", "hạ"
    ]

    claims = []
    sentences = _sentencize_vietnamese(text)

    for sent_text in sentences:
        sent_text = sent_text.strip()
        if not sent_text or sent_text.endswith("?"):
            continue

        sent_doc = nlp(sent_text)
        has_claim_marker = any(marker in sent_text.lower() for marker in vi_claim_markers)

        # Check for subject-verb patterns
        has_subject = any(token.dep_ in ["nsubj", "nsubj:pass"] for token in sent_doc)

        # Check for factual indicators
        factual_indicators = ["năm", "người", "triệu", "tỷ", "phần", "%", "°"]
        has_factual_data = any(ind in sent_text for ind in factual_indicators)

        if (has_claim_marker or has_factual_data) and (has_subject or len(sent_doc) > 5):
            if _looks_like_claim_vietnamese(sent_doc):
                claims.append(sent_text)

    logger.info(f"dependency_claim_extraction_tool completed: {len(claims)} claims found")
    result = {"claims": claims, "method": "dependency", "language": "vi", "dependency_done": True}
    return json.dumps(result, ensure_ascii=False)


def _dependency_extract_english(text: str) -> str:
    """Extract claims from English text using dependency parsing."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    claim_verbs = {
        "say", "claim", "state", "report", "announce", "declare", "assert", "allege",
        "argue", "maintain", "contend", "insist", "affirm", "attest", "testify",
        "reveal", "disclose", "admit", "acknowledge", "confess", "confirm", "deny",
        "show", "prove", "demonstrate", "indicate", "suggest", "imply",
        "note", "observe", "find", "discover", "detect", "notice",
        "emphasize", "stress", "highlight", "underscore", "point out",
        "predict", "forecast", "warn", "caution", "anticipate", "expect",
        "explain", "describe", "characterize", "define", "specify",
        "believe", "think", "consider", "regard", "view",
        "estimate", "calculate", "determine", "assess", "evaluate",
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "contain", "include", "involve"
    }
    claims = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_doc = nlp(sent_text)

        has_claim_verb = any(token.lemma_.lower() in claim_verbs for token in sent_doc)
        has_subject = any(token.dep_ == "nsubj" for token in sent_doc)
        has_object = any(token.dep_ in ["dobj", "pobj", "attr"] for token in sent_doc)

        if has_claim_verb and has_subject and (has_object or len(sent_doc) > 5):
            if not sent_text.endswith("?") and _looks_like_claim(sent_doc):
                claims.append(sent_text)

    logger.info(f"dependency_claim_extraction_tool completed: {len(claims)} claims found")
    result = {"claims": claims, "method": "dependency", "language": "en", "dependency_done": True}
    return json.dumps(result)


@tool()
async def llm_claim_extraction_tool(text: str) -> str:
    """
    Extract claims from text using LLM zero-shot reasoning.
    Uses LLM to identify and extract factual claims.
    Automatically detects language and uses appropriate prompt.

    Args:
        text: Input text to extract claims from

    Returns:
        JSON string with claims list and method
    """
    logger.info(f"[DEBUG] llm_claim_extraction_tool called")

    lang = _detect_language(text)
    logger.info(f"Detected language: {lang}")

    try:
        if lang == "vi":
            return _llm_extract_vietnamese(text)
        return _llm_extract_english(text)
    except Exception as e:
        logger.error(f"Error during LLM extraction: {e}")
        result = {"claims": [], "error": str(e)}
        return json.dumps(result)


def _llm_extract_vietnamese(text: str) -> str:
    """Extract claims from Vietnamese text using LLM."""
    system_prompt = """Bạn là một chuyên gia trong việc trích xuất thông tin thực tế từ văn bản.
Nhiệm vụ của bạn là trích xuất tất cả các tuyên bố có thể kiểm chứng từ văn bản.
Trả về CHỈ một mảng JSON hợp lệ: [{"claim_text": "..."}]
Không có markdown, không có văn bản bổ sung."""

    prompt = f"""Phân tích văn bản sau và trích xuất tất cả các tuyên bố thực tế (factual claims).

Văn bản: {text}

Trích xuất các tuyên bố rõ ràng, có thể kiểm chứng. Các tuyên bố cần:
- Có nội dung khách quan (không phải ý kiến cá nhân)
- Có thể xác minh được bằng dữ liệu, sự kiện
- Chứa thông tin cụ thể (con số, ngày tháng, địa điểm, tên người/tổ chức)

Trả về CHỈ một mảng JSON: [{{"claim_text": "tuyên bố 1"}}, {{"claim_text": "tuyên bố 2"}}]"""

    content = call_llm(prompt, system_prompt=system_prompt, max_tokens=500)

    content = re.sub(r'```json\n?', '', content)
    content = re.sub(r'```\n?', '', content)
    content = content.strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            claims = []
            for item in parsed:
                if isinstance(item, dict) and "claim_text" in item:
                    claims.append(item["claim_text"].strip())
                elif isinstance(item, str):
                    claims.append(item.strip())

            logger.info(f"llm_claim_extraction_tool completed: {len(claims)} claims found")
            result = {"claims": claims, "method": "llm", "language": "vi", "llm_done": True}
            return json.dumps(result, ensure_ascii=False)

        logger.warning("Invalid LLM response format")
        result = {"claims": [], "error": "Invalid LLM response", "language": "vi"}
        return json.dumps(result, ensure_ascii=False)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM response as JSON: {content[:100]}...")
        result = {"claims": [], "error": "JSON parse failed", "language": "vi"}
        return json.dumps(result, ensure_ascii=False)


def _llm_extract_english(text: str) -> str:
    """Extract claims from English text using LLM."""
    prompt = f"""Analyze the following text and extract all distinct factual claims.

Text: {text}

Extract clear, verifiable claims that can be fact-checked.
Return ONLY a valid JSON array: [{{"claim_text": "..."}}, {{"claim_text": "..."}}]
No markdown, no additional text."""

    content = call_llm(
        prompt,
        system_prompt="Extract factual claims and return only valid JSON.",
        max_tokens=500,
    )

    content = re.sub(r'```json\n?', '', content)
    content = re.sub(r'```\n?', '', content)
    content = content.strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            claims = []
            for item in parsed:
                if isinstance(item, dict) and "claim_text" in item:
                    claims.append(item["claim_text"].strip())
                elif isinstance(item, str):
                    claims.append(item.strip())

            logger.info(f"llm_claim_extraction_tool completed: {len(claims)} claims found")
            result = {"claims": claims, "method": "llm", "language": "en", "llm_done": True}
            return json.dumps(result)

        logger.warning("Invalid LLM response format")
        result = {"claims": [], "error": "Invalid LLM response", "language": "en"}
        return json.dumps(result)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM response as JSON: {content[:100]}...")
        result = {"claims": [], "error": "JSON parse failed", "language": "en"}
        return json.dumps(result)


def _looks_like_claim(doc) -> bool:
    """Check if a sentence looks like a factual claim."""
    text = doc.text.strip()
    if text.endswith("?") or text.endswith("!"):
        return False
    if len(doc) < 3:
        return False
    return any(token.pos_ == "VERB" for token in doc)