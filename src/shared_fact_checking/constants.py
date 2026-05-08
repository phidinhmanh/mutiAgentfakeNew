"""Shared constants for fact-checking domains and authority."""

# Authoritative news and government domains for Vietnamese fact-checking
AUTHORITATIVE_DOMAINS = [
    # Tier 1: Government / legal portals
    "gov.vn",
    "chinhphu.vn",
    "baochinhphu.vn",
    "thuvienphapluat.vn",
    "quochoi.vn",
    "vanban.chinhphu.vn",
    # Tier 2: Major national press
    "vnexpress.net",
    "tuoitre.vn",
    "thanhnien.vn",
    "vietnamnet.vn",
    "nhandan.vn",
    "vietnamplus.vn",
    "laodong.vn",
    "dantri.com.vn",
    "vtv.vn",
    "giaoducthoidai.vn",
    "congthuong.vn",
    "nongnghiepmoitruong.vn",
    "baotintuc.vn",
]

# Domains considered untrustworthy or satire
UNTRUSTED_DOMAINS = [
    "cuoichut.com",
    "vinhphuc.work",  # example of suspicious domain
]

# Thresholds
NUMERIC_DISCREPANCY_THRESHOLD = 0.01
CACHE_TTL_SECONDS = 600
EVIDENCE_RETRIEVAL_TIMEOUT = 3.0
EVIDENCE_RETRIEVAL_TOP_K = 2
SEARCH_CACHE_TTL = 300
DDG_REQUEST_TIMEOUT = 10
DDG_SHORT_RESPONSE_THRESHOLD = 5000
RETRY_MAX_RETRIES = 5
RETRY_BASE_DELAY = 30
LOW_CONFIDENCE_RETRY_THRESHOLD = 0.7
VETO_CONFIDENCE_THRESHOLD_SINGLE = 0.5
VETO_CONFIDENCE_THRESHOLD_MULTI = 0.7
GOV_AUTHORITY_WEIGHT = 3.5
NEWS_AUTHORITY_WEIGHT = 2.0
NEGATION_GUARD_THRESHOLD = 0.85
NUMERIC_MISMATCH_THRESHOLD = 0.10
GOV_CONFIDENCE_BOOST = 0.15

# API
API_THREAD_POOL_WORKERS = 4

# Domains
TRUSTED_TIER_PREFIXES = tuple(AUTHORITATIVE_DOMAINS)
STRICT_EXTRACTION_DOMAINS = AUTHORITATIVE_DOMAINS + [
    "vi.wikipedia.org",
    "en.wikipedia.org",
]

NEGATION_KEYWORDS = {
    "không",
    "chưa",
    "chẳng",
    "sai",
    "bác bỏ",
    "bác",
    "stop",
    "deny",
    "refute",
    "ngừng",
    "dừng",
    "cấm",
    "phủ nhận",
    "từ chối",
    "không có",
    "không phải",
    "không đúng",
    "không chính xác",
    "không được",
    "chống lại",
    "ngược lại",
    "phản đối",
    "tẩy chay",
}

NEGATION_TRIGGER_MODIFIERS = {
    "tuy nhiên",
    "nhưng",
    "dù",
    "mặc dù",
    "tuy",
    "trái lại",
    "thực tế là",
}

# Site filters for search engines (DuckDuckGo, etc.)
NEWS_SITE_FILTERS = [
    "vnexpress.net",
    "thanhnien.vn",
    "tuoitre.vn",
    "vietnamnet.vn",
    "nhandan.vn",
]

# Domain credibility weights for score calibration
DOMAIN_CREDIBILITY_WEIGHTS = {
    "vnexpress.net": 1.0,
    "thanhnien.vn": 1.0,
    "tuoitre.vn": 1.0,
    "vietnamnet.vn": 1.0,
    "nhandan.vn": 0.95,
    "laodong.vn": 0.9,
    "dantri.com.vn": 0.9,
    "vtv.vn": 1.0,
    "vietnamplus.vn": 1.0,
    "baotintuc.vn": 0.95,
}
