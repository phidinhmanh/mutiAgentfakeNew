"""Query helpers for evidence retrieval."""

from __future__ import annotations

import logging
import re

from shared_fact_checking.constants import AUTHORITATIVE_DOMAINS, NEWS_SITE_FILTERS

logger = logging.getLogger("EvidenceRetriever.Agent")

NEWS_SITE_FILTERS_TOP = NEWS_SITE_FILTERS
STRICT_DOMAINS = AUTHORITATIVE_DOMAINS + [
    "vi.wikipedia.org",
    "en.wikipedia.org",
]

CURRENT_EVENT_MARKERS = [
    "năm nay",
    "mới đây",
    "vừa qua",
    "hiện nay",
    "đang diễn ra",
    "hôm nay",
    "gần đây",
    "2025",
    "2026",
    "sắp tới",
    "cuối tuần",
    "tuần này",
    "tháng này",
    "gần nhất",
    "lần đầu",
    "lần thứ",
    "chưa từng có",
]

RECENT_EVENT_MARKERS = [
    "2024",
    "2023",
    "năm ngoái",
    "năm trước",
]

OLDER_EVENT_MARKERS = [
    "2022",
    "2021",
    "2020",
    "cách đây",
    "đã qua",
    "hồi tưởng",
    "giải nghệ",
    "đã mất",
    "đã qua đời",
]

HISTORICAL_MARKERS = [
    "lịch sử",
    "thế kỷ",
    "kỷ niệm",
    "ngày thành lập",
    "truyền thống",
    "xưa",
    "cổ",
    "trước đây",
]

QUERY_STOPWORDS = {
    "đã",
    "đang",
    "sẽ",
    "có",
    "được",
    "là",
    "và",
    "của",
    "cho",
    "với",
    "theo",
    "trong",
    "năm",
    "tháng",
    "ngày",
    "giờ",
    "phút",
    "hay",
    "hoặc",
    "này",
    "đó",
    "kia",
    "ở",
    "tại",
    "về",
    "ra",
    "vào",
    "từ",
    "đến",
    "vì",
    "những",
    "các",
    "một",
    "hai",
    "ba",
    "bốn",
    "sáu",
    "bảy",
    "tám",
    "hôm",
    "nay",
    "qua",
    "bởi",
    "nên",
    "để",
    "tuy",
    "nhưng",
    "mà",
    "vẫn",
    "còn",
    "chỉ",
    "rằng",
    "khi",
    "nếu",
    "thì",
    "do",
    "cũng",
    "đều",
    "tất",
    "cả",
    "mọi",
    "ai",
    "gì",
    "đâu",
    "hiện",
    "mới",
    "vừa",
    "lại",
    "luôn",
    "mãi",
    "bao",
    "như",
    "vậy",
    "thế",
    "nào",
    "sao",
    "hơn",
    "kém",
    "nhất",
    "không",
    "phải",
    "có phải",
    "đúng không",
    "sao không",
    "Thủ tướng",
    "Bộ trưởng",
    "Chủ tịch",
    "Tổng thư ký",
    "Thị trường",
}

BROAD_QUERY_STOPWORDS = {
    "đã",
    "đang",
    "sẽ",
    "có",
    "được",
    "là",
    "và",
    "của",
    "cho",
    "với",
    "theo",
    "trong",
    "năm",
    "tháng",
    "ngày",
    "giờ",
    "phút",
    "hay",
    "hoặc",
    "này",
    "đó",
    "kia",
    "ở",
    "tại",
    "về",
    "ra",
    "vào",
    "từ",
    "đến",
    "vì",
    "những",
    "các",
    "một",
    "hai",
    "ba",
    "bốn",
    "sáu",
    "bảy",
    "tám",
    "trên",
    "dưới",
    "trước",
    "sau",
    "khi",
    "nếu",
    "thì",
    "do",
    "nên",
    "để",
    "tuy",
    "nhưng",
    "mà",
    "vẫn",
    "còn",
    "chỉ",
    "rằng",
}

SYNONYMS = {
    "ông": ["ông", "anh", "ôngng"],
    "bà": ["bà", "chị", "bà"],
    "thủ tướng": ["thủ tướng", "tổng bí thư", "chính phủ"],
    "chủ tịch": ["chủ tịch", "trưởng ban", "giám đốc"],
    "bộ trưởng": ["bộ trưởng", "thứ trưởng", "bộ"],
    "nghỉ hưu": ["nghỉ hưu", "về hưu", "nghỉ việc"],
    "học bổng": ["học bổng", "học phí", "tài trợ"],
    "đầu tư": ["đầu tư", "tài trợ", "tài chính"],
    "phát biểu": ["phát biểu", "nói", "tuyên bố"],
    "quyết định": ["quyết định", "ban hành", "phê duyệt"],
}

BIAS_WORDS = {
    "scandal",
    "bê bối",
    "giật gân",
    "gây sốc",
    "không thể tin",
    "bất ngờ",
    "đáng kinh ngạc",
    "kinh hoàng",
    "sốc",
    "chấn động",
    "gây tranh cãi",
    "nóng",
    "nổi đình nổi đám",
    "đình đám",
    "rúng động",
    "choáng váng",
    "khủng khiếp",
    "thảm họa",
    "tai tiếng",
    "nhục nhã",
    "thê thảm",
}

ACTION_PATTERNS = [
    r"(?:cấp|phát|hủy|xóa|bổ nhiệm|tặng|trao|kỷ luật|cảnh cáo|khiển trách|khai trừ)\s+(?:mã|phép|giấy|chứng|tài|khoản|quyết định|đảng|tổ chức)",
    r"(?:nâng cấp|hạ cấp|xây dựng|khởi công|khánh thành|kích hoạt|xác định)\s+\w+(?:\s+\w+)?",
    r"mã\s+(?:định danh|tạm|phẩm|bảo|hành)",
    r"tài\s+khoản\s+(?:định danh|điện tử|cá nhân)",
    r"(?:kỷ niệm|đánh dấu|mừng)\s+\d+\s+\w+",
    r"(?:trả lời|trả lời\s+câu\s+hỏi|giải đáp|khẳng định|phát biểu|tuyên bố|nói)\s+\w+",
    r"(?:sáp nhập|tách|bỏ|thành lập|giải thể)\s+\w+(?:\s+\w+)?",
    r"(?:họp báo|buổi\s+họp|kỳ\s+họp|hội\s+nghị)\s+\w+",
    r"(?:xử lý|xử phạt|phạt)\s+\d+\s+(?:trường hợp|triệu|tỷ)",
]


def filter_urls_by_strict_domains(urls: list[str]) -> list[str]:
    return [url for url in urls if any(domain in url.lower() for domain in STRICT_DOMAINS)]


def generate_news_site_query(keyword_query: str) -> str:
    news_sites = NEWS_SITE_FILTERS_TOP[:2]
    site_part = " OR ".join(f"site:{site}" for site in news_sites)
    return f"{keyword_query} ({site_part})"


def _append_inferred_year(numbers: list[str], claim: str) -> None:
    lowered = claim.lower()
    if any(marker in lowered for marker in CURRENT_EVENT_MARKERS):
        numbers.append("2025")
    elif any(marker in lowered for marker in RECENT_EVENT_MARKERS):
        numbers.append("2024")
    elif any(marker in lowered for marker in OLDER_EVENT_MARKERS):
        numbers.append("2023")
    elif any(marker in lowered for marker in HISTORICAL_MARKERS):
        numbers.append("2023")
    else:
        numbers.append("2025")


def generate_keyword_query(claim: str) -> str:
    numbers = re.findall(r"\b\d{4}\b", claim)
    dates_in_claim = re.findall(r"\b\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b", claim)
    numbers.extend(dates_in_claim)

    short_date_found = any(len(re.split(r"[/.-]", date)) == 2 for date in dates_in_claim)
    year_found = any(len(number) == 4 and number.startswith("20") for number in numbers)
    if short_date_found and not year_found:
        numbers.append("2023")
    if not year_found:
        _append_inferred_year(numbers, claim)

    numbers += re.findall(r"\b\d+(?:[.,]\d+)*%?\b", claim)
    quotes = re.findall(r'"([^"]+)"', claim)
    proper_nouns = re.findall(r"\b[A-ZÀ-Ỹ][a-zà-ỹ]*(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]*){0,3}\b", claim)

    keywords: list[str] = []
    for noun in proper_nouns:
        noun_clean = noun.strip()
        noun_lower = noun_clean.lower()
        if len(noun_clean) > 2 and noun_lower not in QUERY_STOPWORDS:
            if not any(stopword in noun_lower for stopword in QUERY_STOPWORDS if len(stopword) > 3):
                keywords.append(noun_clean)

    action_phrases: list[str] = []
    for pattern in ACTION_PATTERNS:
        for match in re.finditer(pattern, claim):
            phrase = match.group(0).strip()
            if len(phrase) > 4:
                action_phrases.append(phrase)

    parts: list[str] = []
    for quote in quotes[:2]:
        if len(quote) > 2:
            parts.append(f'"{quote}"')

    if action_phrases:
        best_action = max(action_phrases, key=len)
        parts.append(best_action)

    important_numbers: list[str] = []
    for num in numbers:
        if re.match(r"^\d{4}$", num) or "000" in num or len(num) > 3 or "/" in num or "-" in num:
            important_numbers.append(num)
    parts.extend(important_numbers[:3])
    parts.extend(keywords[:4])

    non_stop_tokens = [token.strip(',.():;!?"').lower() for token in claim.split() if token.strip(',.():;!?"') and token.strip(',.():;!?"').lower() not in QUERY_STOPWORDS and len(token.strip(',.():;!?"')) >= 4]
    if len(parts) <= 1 and non_stop_tokens:
        lexical_fallback: list[str] = []
        for token in non_stop_tokens:
            if token not in lexical_fallback:
                lexical_fallback.append(token)
            if len(lexical_fallback) >= 5:
                break
        if not action_phrases and not keywords:
            parts = lexical_fallback + important_numbers[:2]

    if not parts:
        return claim[:50] if len(claim) > 50 else claim

    result = " ".join(parts)
    logger.info("Keyword query generated: '%s'", result)
    return result


def generate_number_focused_query(claim: str) -> str:
    numbers = re.findall(r"\b\d+(?:[.,]\d+)*\b", claim)
    dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", claim)
    quotes = re.findall(r'"([^"]+)"', claim)
    if not numbers and not dates and not quotes:
        return claim

    parts: list[str] = []
    if quotes:
        parts.extend(quotes[:2])
    if dates:
        parts.extend(dates[:2])
    if numbers:
        sig_numbers = [n for n in numbers if len(n.replace(",", "").replace(".", "")) >= 3]
        parts.extend(sig_numbers[:3])
    if not parts:
        return claim

    focus = " ".join(parts[:4])
    if "Việt" not in claim and "VN" not in claim:
        return f"{focus} Vietnam"
    return focus


def generate_broader_query(claim: str) -> str:
    words = claim.lower().split()
    key_words = [word for word in words if word not in BROAD_QUERY_STOPWORDS and len(word) >= 3]

    expanded = list(key_words)
    for word in key_words:
        if word in SYNONYMS:
            expanded.extend(SYNONYMS[word][:2])

    unique = list(dict.fromkeys(expanded))[:6]
    query = " ".join(unique)
    if "việt" not in claim.lower() and "vn" not in claim.lower():
        query += " Vietnam"
    return query if query else claim


def strip_bias_words(claim: str) -> str:
    words = claim.split()
    stripped = [word for word in words if word.lower() not in BIAS_WORDS]
    return " ".join(stripped)


def generate_contradiction_queries(claim: str) -> list[str]:
    q1 = generate_keyword_query(claim)
    neutral = strip_bias_words(claim)
    subject = neutral.split(",")[0].split(";")[0].split(".")[0]
    if len(subject) > 60:
        subject = subject[:60]
    q2 = generate_keyword_query(f"{subject} thực tế")
    q3 = generate_keyword_query(f"{subject} xác minh sự thật")
    logger.info(
        "[V12 ULTRA] Contradiction queries: Q1(confirm)='%s...' Q2(counter)='%s...' Q3(verify)='%s...'",
        q1[:50],
        q2[:50],
        q3[:50],
    )
    return [q1, q2, q3]


def extract_context_years(context: str | None) -> list[str]:
    if not context:
        return []
    years = re.findall(r"\b20\d{2}\b", context)
    ordered: list[str] = []
    for year in years:
        if year not in ordered:
            ordered.append(year)
    return ordered[:3]


def extract_context_topic_terms(context: str | None) -> list[str]:
    if not context:
        return []
    matches = re.findall(r"\b[A-ZÀ-Ỹ][A-ZÀ-Ỹa-zà-ỹ]*(?:\s+[A-ZÀ-Ỹ][A-ZÀ-Ỹa-zà-ỹ]*){0,3}\b", context)
    topic_terms: list[str] = []
    for match in matches:
        candidate = match.strip()
        if len(candidate) <= 1:
            continue
        if len(candidate.split()) == 1 and len(candidate) <= 3:
            continue
        if candidate not in topic_terms:
            topic_terms.append(candidate)
        if len(topic_terms) >= 4:
            break
    return topic_terms


def inject_context_anchors(query: str, ground_truth_evidence: str | None) -> str:
    if not ground_truth_evidence:
        return query

    context_years = extract_context_years(ground_truth_evidence)
    topic_terms = extract_context_topic_terms(ground_truth_evidence)
    augmented = query
    if topic_terms:
        augmented = f"{' '.join(topic_terms[:2])} {augmented}"
    if context_years:
        for year in context_years[:2]:
            if year not in augmented:
                augmented = f"{year} {augmented}"

    logger.info(
        "[V13] Context anchors injected: years=%s topic_terms=%s",
        context_years[:2],
        topic_terms[:2],
    )
    return augmented
