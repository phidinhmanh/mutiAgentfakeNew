"""
Claim Extractor Tools - Tools for NLP-based claim extraction.

Tools used by the Claim Extractor ReAct Agent:
- NER Tool: Named Entity Recognition
- Dependency Parsing Tool: Sentence structure analysis
- LLM Tool: Zero-shot reasoning
"""

from __future__ import annotations

import json
import logging
import re

from dotenv import load_dotenv
from langchain_core.tools import tool

from trust_agents.agents.claim_nlp_utils import (
    detect_language as _detect_language,
    ENGLISH_CLAIM_VERBS,
    fallback_claim_sentences as _fallback_claim_sentences,
    looks_like_claim_english as _looks_like_claim_english,
    looks_like_claim_vietnamese as _looks_like_claim_vietnamese,
    load_spacy_model as _load_spacy_model,
    sentencize_vietnamese as _sentencize_vietnamese,
    VIETNAMESE_CLAIM_MARKERS,
)
from trust_agents.llm.llm_helpers import call_llm

load_dotenv()
logger = logging.getLogger("TRUST_agents.agents.claim_extractor_tools")
logger.propagate = True


@tool()
async def ner_claim_extraction_tool(text: str) -> str:
    """Extract claims using Named Entity Recognition.

    Identifies sentences containing named entities (people, organizations,
    locations, etc.). Supports both English and Vietnamese.
    """
    logger.info("[DEBUG] ner_claim_extraction_tool called")
    lang = _detect_language(text)
    logger.info("Detected language: %s", lang)
    try:
        return _ner_extract_vietnamese(text) if lang == "vi" else _ner_extract_english(text)
    except Exception as e:
        logger.error("Error during NER extraction: %s", e)
        return json.dumps({"claims": [], "error": str(e)})


def _ner_extract_vietnamese(text: str) -> str:
    nlp = _load_spacy_model("xx_ent_wiki_sm", "xx_sm")
    if nlp is None:
        claims = _fallback_claim_sentences(text)
        logger.info("ner_claim_extraction_tool completed (heuristic): %d claims", len(claims))
        return json.dumps({"claims": claims, "method": "ner", "language": "vi", "ner_done": False, "fallback": "heuristic"}, ensure_ascii=False)

    claims = []
    for sent in _sentencize_vietnamese(text):
        sent_doc = nlp(sent.strip())
        entities = [
            ent.text
            for ent in sent_doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "EVENT"]
        ]
        if entities and _looks_like_claim_vietnamese(sent_doc):
            claims.append(sent.strip())

    logger.info("ner_claim_extraction_tool completed: %d claims found", len(claims))
    return json.dumps({"claims": claims, "method": "ner", "language": "vi", "ner_done": True}, ensure_ascii=False)


def _ner_extract_english(text: str) -> str:
    nlp = _load_spacy_model("en_core_web_sm", "xx_ent_wiki_sm", "xx_sm")
    if nlp is None:
        claims = _fallback_claim_sentences(text)
        logger.info("ner_claim_extraction_tool completed (heuristic): %d claims", len(claims))
        return json.dumps({"claims": claims, "method": "ner", "language": "en", "ner_done": False, "fallback": "heuristic"})

    doc = nlp(text)
    claims = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_doc = nlp(sent_text)
        entities = [
            ent.text
            for ent in sent_doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "MONEY"]
        ]
        if entities and _looks_like_claim_english(sent_doc):
            claims.append(sent_text)

    logger.info("ner_claim_extraction_tool completed: %d claims found", len(claims))
    return json.dumps({"claims": claims, "method": "ner", "language": "en", "ner_done": True})


@tool()
async def dependency_claim_extraction_tool(text: str) -> str:
    """Extract claims using dependency parsing.

    Identifies claim patterns: subject-verb-object structures with factual content.
    Supports both English and Vietnamese.
    """
    logger.info("[DEBUG] dependency_claim_extraction_tool called")
    lang = _detect_language(text)
    logger.info("Detected language: %s", lang)
    try:
        return _dependency_extract_vietnamese(text) if lang == "vi" else _dependency_extract_english(text)
    except Exception as e:
        logger.error("Error during dependency parsing: %s", e)
        return json.dumps({"claims": [], "error": str(e)})


def _dependency_extract_vietnamese(text: str) -> str:
    nlp = _load_spacy_model("xx_ent_wiki_sm", "xx_sm", "en_core_web_sm")
    if nlp is None:
        claims = _fallback_claim_sentences(text)
        logger.info("dependency_claim_extraction_tool completed (heuristic): %d claims", len(claims))
        return json.dumps({"claims": claims, "method": "dependency", "language": "vi", "dependency_done": False, "fallback": "heuristic"}, ensure_ascii=False)

    claims = []
    for sent_text in _sentencize_vietnamese(text):
        sent_text = sent_text.strip()
        if not sent_text or sent_text.endswith("?"):
            continue
        sent_doc = nlp(sent_text)
        has_claim_marker = any(marker in sent_text.lower() for marker in VIETNAMESE_CLAIM_MARKERS)
        has_subject = any(token.dep_ in ["nsubj", "nsubj:pass"] for token in sent_doc)
        has_factual_data = any(ind in sent_text for ind in ["năm", "người", "triệu", "tỷ", "phần", "%", "°"])
        if (has_claim_marker or has_factual_data) and (has_subject or len(sent_doc) > 5):
            if _looks_like_claim_vietnamese(sent_doc):
                claims.append(sent_text)

    logger.info("dependency_claim_extraction_tool completed: %d claims found", len(claims))
    return json.dumps({"claims": claims, "method": "dependency", "language": "vi", "dependency_done": True}, ensure_ascii=False)


def _dependency_extract_english(text: str) -> str:
    nlp = _load_spacy_model("en_core_web_sm", "xx_ent_wiki_sm", "xx_sm")
    if nlp is None:
        claims = _fallback_claim_sentences(text)
        logger.info("dependency_claim_extraction_tool completed (heuristic): %d claims", len(claims))
        return json.dumps({"claims": claims, "method": "dependency", "language": "en", "dependency_done": False, "fallback": "heuristic"})

    doc = nlp(text)
    claims = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_doc = nlp(sent_text)
        has_claim_verb = any(token.lemma_.lower() in ENGLISH_CLAIM_VERBS for token in sent_doc)
        has_subject = any(token.dep_ == "nsubj" for token in sent_doc)
        has_object = any(token.dep_ in ["dobj", "pobj", "attr"] for token in sent_doc)
        if has_claim_verb and has_subject and (has_object or len(sent_doc) > 5):
            if not sent_text.endswith("?") and _looks_like_claim_english(sent_doc):
                claims.append(sent_text)

    logger.info("dependency_claim_extraction_tool completed: %d claims found", len(claims))
    return json.dumps({"claims": claims, "method": "dependency", "language": "en", "dependency_done": True})


@tool()
async def llm_claim_extraction_tool(text: str) -> str:
    """Extract claims using LLM zero-shot reasoning."""
    logger.info("[DEBUG] llm_claim_extraction_tool called")
    lang = _detect_language(text)
    logger.info("Detected language: %s", lang)
    try:
        return _llm_extract_vietnamese(text) if lang == "vi" else _llm_extract_english(text)
    except Exception as e:
        logger.error("Error during LLM extraction: %s", e)
        return json.dumps({"claims": [], "error": str(e)})


def _llm_extract_vietnamese(text: str) -> str:
    system_prompt = """Bạn là một chuyên gia trong việc trích xuất thông tin thực tế từ văn bản.
Trả về CHỈ một mảng JSON hợp lệ: [{"claim_text": "..."}]
Không có markdown, không có văn bản bổ sung."""
    prompt = f"""Phân tích văn bản sau và trích xuất tất cả các tuyên bố thực tế có thể kiểm chứng.

Văn bản: {text}

Trích xuất các tuyên bố rõ ràng, có thể xác minh được. Trả về CHỈ một mảng JSON: [{{"claim_text": "tuyên bố 1"}}, {{"claim_text": "tuyên bố 2"}}]"""

    content = _strip_json_markers(call_llm(prompt, system_prompt=system_prompt, max_tokens=500))
    return _parse_llm_claims(content, lang="vi")


def _llm_extract_english(text: str) -> str:
    prompt = f"""Analyze the following text and extract all distinct factual claims.
Return ONLY a valid JSON array: [{{"claim_text": "..."}}, {{"claim_text": "..."}}]
No markdown, no additional text.

Text: {text}"""
    content = _strip_json_markers(
        call_llm(prompt, system_prompt="Extract factual claims and return only valid JSON.", max_tokens=500)
    )
    return _parse_llm_claims(content, lang="en")


def _strip_json_markers(content: str) -> str:
    content = re.sub(r"```json\n?", "", content)
    content = re.sub(r"```\n?", "", content)
    return content.strip()


def _parse_llm_claims(content: str, lang: str) -> str:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            claims = []
            for item in parsed:
                if isinstance(item, dict) and "claim_text" in item:
                    claims.append(item["claim_text"].strip())
                elif isinstance(item, str):
                    claims.append(item.strip())
            logger.info("llm_claim_extraction_tool completed: %d claims found", len(claims))
            return json.dumps({"claims": claims, "method": "llm", "language": lang, "llm_done": True}, ensure_ascii=False)
        logger.warning("Invalid LLM response format")
        return json.dumps({"claims": [], "error": "Invalid LLM response", "language": lang}, ensure_ascii=False)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON: %s...", content[:100])
        return json.dumps({"claims": [], "error": "JSON parse failed", "language": lang}, ensure_ascii=False)