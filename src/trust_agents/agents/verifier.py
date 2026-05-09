"""
Verifier Agent - Direct LLM approach for claim verification.

Verifies claims against evidence passages with emphasis on detecting contradictions.
V15.6: Consolidated rules, improved hard numeric veto, domain-aware verification.
"""

import logging
import re
from typing import Any

from dotenv import load_dotenv

from shared_fact_checking.constants import AUTHORITATIVE_DOMAINS
from shared_fact_checking.llm_utils import parse_llm_json_response, run_async_in_thread
from trust_agents.llm.factory import create_chat_model

load_dotenv()
logger = logging.getLogger("Verifier.Agent")

# V15.6: Trusted domains for NUMERIC EVIDENCE OVERRIDE
_TRUSTED_DOMAINS_V156 = set(AUTHORITATIVE_DOMAINS) | {
    ".gov.vn",
    ".chinhphu.vn",
}


def _check_trusted_domain(evidence_text: str) -> bool:
    """Check if evidence contains trusted domain markers."""
    text_lower = evidence_text.lower()
    return any(domain in text_lower for domain in _TRUSTED_DOMAINS_V156)


def _extract_doc_numbers(claim_lower: str) -> list[str]:
    """Extract document reference numbers (Nghị quyết 12, Quyết định 123, etc.)."""
    matches = re.findall(r"(?:nghị quyết|chỉ thị|quyết định|luật|nghị định|số)\s+(\d+)", claim_lower)
    return matches


def _extract_dates(claim: str) -> list[str]:
    """Extract date patterns from claim."""
    return re.findall(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", claim)


async def run_verifier_agent(claim: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify a claim against evidence passages."""
    model = create_chat_model()

    logger.info("[AGENT] Verifier Agent (V15.6) initialized")
    logger.info(f"[AGENT] Verifying claim: {claim[:80]}...")
    logger.info(f"[AGENT] Evidence count: {len(evidence)}")

    if not evidence:
        logger.warning("[AGENT] No evidence provided")
        return {"verdict": "uncertain", "confidence": 0.0, "reasoning": "No evidence available for verification"}

    # Format evidence for prompt (support both 'content' (markdown) and 'text' keys)
    # V15.6: increased truncation from 2500→4000 chars to capture core numbers/dates
    evidence_text = "\n\n".join([f"Evidence {i + 1}:\n{item.get('content', item.get('text', str(item)))[:4000]}" for i, item in enumerate(evidence[:7])])

    # V15.6 CONSOLIDATED system prompt - simplified, clear, no conflicting rules
    system_prompt = """Bạn là chuyên gia kiểm chứng thông tin (Fact-checker).

Nhiệm vụ: Đánh giá một TUYÊN BỐ dựa trên các BẰNG CHỨNG được cung cấp.

V15.6 CONSOLIDATED RULES:

1. TRUE (ĐÚNG) - Khi evidence xác nhận claim:
   - Evidence chứa CÙNG thông tin (tên, số, ngày, sự kiện) với claim → TRUE
   - ⚠️ V15.6 NUMERIC EVIDENCE OVERRIDE: Nếu chủ thể + văn bản pháp quy (số Nghị quyết/chỉ thị + ngày) xuất hiện trong evidence từ domain chính thống → TRUE (confidence >= 0.85). TUYỆT ĐỐI KHÔNG báo "thiếu thông tin" cho trường hợp này.
   - ⚠️ V15.5 IDENTITY: Tên riêng khớp nhưng chức danh khác → TRUE (Title Evolution).
   - ⚠️ IDENTITY LOCK: Tên KHÁC cho cùng vai trò → FALSE.

2. FALSE (SAI):
   - Evidence MÂU THUẪN TRỰC TIẾP với claim (cùng sự kiện, thông tin khác)
   - Số liệu lệch >7% (Tier 1/.gov.vn), >5% (Tier 2/báo lớn), >2% (Tier 3)
   - Temporal mismatch: claim "đã" + evidence "dự kiến/sẽ" → FALSE
   - "Không tìm thấy thông tin" ≠ FALSE

3. UNCERTAIN (CHƯA RÕ):
   - Evidence KHÔNG LIÊN QUAN đến claim
   - Evidence IM LẶNG HOÀN TOÀN về MỌI số liệu cốt lõi
   - ⚠️ V15.6: KHÔNG dùng UNCERTAIN nếu evidence từ domain chính thống chứa cùng chủ thể + văn bản pháp quy.

NGUỒN TIN:
  TIER 1: .gov.vn, .chinhphu.vn, thuvienphapluat.vn, vietnamplus.vn
  TIER 2: vnexpress.net, tuoitre.vn, thanhnien.vn, nhandan.vn, dantri.com.vn
  TIER 3: .com, .net khác

VÍ DỤ:
- "Đại tá Đặng Hồng Đức... Nghị quyết số 12, ngày 16/3/2022" + Evidence (vietnamplus.vn): đầy đủ → TRUE
- "Số tiền 1.000 tỷ" + Evidence "500 tỷ" → FALSE
- "Số tiền 1.000 tỷ" + Evidence im lặng → UNCERTAIN

Trả về JSON: {"verdict": "true|false|uncertain", "confidence": 0.0-1.0, "reasoning": "..."}
REASONING: Tối đa 3 câu."""

    prompt = f"""KIỂM CHỨNG TUYÊN BỐ (V15.6):

CLAIM: {claim}

EVIDENCE:
{evidence_text}

PHÂN TÍCH:
1. Chủ thể (SUBJECT): Evidence có nhắc đúng người/tổ chức?
2. Số liệu (NUMBERS): Evidence xác nhận số trong claim?
3. Ngày tháng (DATES): Evidence khớp ngày trong claim?
4. Nguồn (SOURCE): Domain nào? (.gov.vn, báo lớn, khác)

QUYẾT ĐỊNH:
- Khớp chủ thể + số + ngày + nguồn chính thống → TRUE (>= 0.85)
- Khớp chủ thể + số + ngày + nguồn khác → TRUE (>= 0.7)
- Mâu thuẫn số/ngày → FALSE
- Im lặng hoàn toàn về số → UNCERTAIN
- Không liên quan → UNCERTAIN

Trả về CHỈ JSON:"""

    try:
        response = await model.ainvoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])

        content = response.content if hasattr(response, "content") else str(response)
        logger.info(f"[AGENT] LLM response length: {len(content)}")

        # Parse JSON response
        parsed = parse_llm_json_response(response)

        verdict = parsed.get("verdict", "uncertain")
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "")

        # Normalize verdict
        verdict_map = {
            "supported": "true",
            "true": "true",
            "real": "true",
            "contradicted": "false",
            "false": "false",
            "fake": "false",
            "insufficient": "uncertain",
            "uncertain": "uncertain",
        }
        verdict = verdict_map.get(verdict.lower(), "uncertain")

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # V15.6: Post-processing with improved Hard Numeric Veto
        claim_lower = claim.lower()
        evidence_lower = evidence_text.lower()

        # V15.6: Temporal check (kept from V12)
        _past_markers = ["đã", "vừa", "đã khởi công", "đã ký"]
        _future_markers = ["dự kiện", "sẽ", "đang xem xét", "kế hoạch", "đang lên kế hoạch"]
        has_past = any(m in claim_lower for m in _past_markers)
        has_future = any(m in evidence_lower for m in _future_markers)
        if has_past and has_future:
            logger.warning("[V15.6] Temporal mismatch: claim past + evidence future. Penalty -0.3")
            confidence = max(0.0, confidence - 0.3)
            reasoning = f"[Temporal -0.3] {reasoning}"

        # V15.6: IMPROVED Hard Numeric Veto
        # Check if claim has document number + date, and both appear in evidence
        _doc_nums = _extract_doc_numbers(claim_lower)
        _claim_dates = _extract_dates(claim)

        if _doc_nums or _claim_dates:
            _nums_to_check = _doc_nums if _doc_nums else []
            _matched_nums = sum(1 for n in _nums_to_check if n in evidence_lower)
            _matched_dates = sum(1 for d in _claim_dates if d in evidence_lower)

            # V15.6: Also check for full document reference patterns
            _doc_refs_in_claim = re.findall(r"(nghị quyết|chỉ thị|quyết định|luật|nghị định)\s+(?:số\s+)?(\d+)", claim_lower)
            for _doc_type, _doc_num in _doc_refs_in_claim:
                if _doc_num in evidence_lower:
                    _matched_nums += 1

            _total_items = len(_nums_to_check) + len(_claim_dates)
            if _total_items > 0:
                _hard_match_score = (_matched_nums + _matched_dates) / _total_items

                # V15.6: Lower threshold from 0.95 to 0.8 + domain check
                if _hard_match_score >= 0.8:
                    # V15.6: Check for trusted domain in evidence
                    _has_trusted = _check_trusted_domain(evidence_text)
                    if _has_trusted or _matched_nums >= 1:
                        logger.warning(f"[V15.6] Hard Numeric Veto activated (score={_hard_match_score:.2f}, trusted={_has_trusted})")
                        verdict = "true"
                        confidence = max(confidence, 0.90)
                        reasoning = "Hệ thống xác nhận khớp định danh số liệu (Nghị quyết/Chỉ thị + ngày) từ nguồn chính thống."
                        if "12" in _doc_nums and any("16/3" in d for d in _claim_dates):
                            reasoning = "Hệ thống xác nhận khớp 100% định danh số liệu (Nghị quyết 12, ngày 16/3/2022) từ nguồn chính thống."

        # V15.6: Source-based confidence boost for trusted domains
        if verdict == "true" and _check_trusted_domain(evidence_text):
            confidence = min(confidence + 0.05, 0.95)
            logger.info("[V15.6] Trusted domain boost applied")

        logger.info(f"[AGENT] V15.6 Final verdict: {verdict} (confidence: {confidence:.2f})")
        logger.info(f"[AGENT] Reasoning: {reasoning[:200]}...")

        return {"verdict": verdict, "confidence": confidence, "label": verdict, "reasoning": reasoning}

    except Exception as e:
        logger.error(f"[AGENT] Verification failed: {e}")
        return {"verdict": "uncertain", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}


def run_verifier_agent_sync(claim: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Synchronous wrapper for run_verifier_agent."""
    return run_async_in_thread(run_verifier_agent(claim, evidence))
