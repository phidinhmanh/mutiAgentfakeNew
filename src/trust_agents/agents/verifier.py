"""
Verifier Agent - Direct LLM approach for claim verification.

Verifies claims against evidence passages with emphasis on detecting contradictions.
"""

import json
import logging
from typing import Any

from dotenv import load_dotenv

from trust_agents.llm.factory import create_chat_model

load_dotenv()
logger = logging.getLogger("Verifier.Agent")


async def run_verifier_agent(claim: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify a claim against evidence passages."""
    model = create_chat_model()

    logger.info("[AGENT] Verifier Agent (Direct LLM) initialized")
    logger.info(f"[AGENT] Verifying claim: {claim[:80]}...")
    logger.info(f"[AGENT] Evidence count: {len(evidence)}")

    if not evidence:
        logger.warning("[AGENT] No evidence provided")
        return {"verdict": "uncertain", "confidence": 0.0, "reasoning": "No evidence available for verification"}

    # Format evidence for prompt (support both 'content' (markdown) and 'text' keys)
    evidence_text = "\n\n".join(
        [
            f"Evidence {i + 1}:\n{item.get('content', item.get('text', str(item)))[:1500]}"
            for i, item in enumerate(evidence[:4])
        ]
    )

    # Strict verification: require EXPLICIT confirmation for TRUE, no inference
    # Core-vs-secondary number distinction (commentary — logic lives in prompt below)
    # - CORE: số tiền gian lận, số nạn nhân, tỷ lệ chính trị, số người chết trong thảm họa
    # - SECONDARY: năm, tháng, địa điểm phụ, số điện thoại, số lượng nhỏ trong context rộng
    system_prompt = """Bạn là chuyên gia kiểm chứng thông tin (Fact-checker).

Nhiệm vụ: Đánh giá một TUYÊN BỐ dựa trên các BẰNG CHỨNG được cung cấp.

QUY TẮC ĐÁNH GIÁ BẮT BUỘC:

1. TRUE (ĐÚNG) - ƯU TIÊN CHO TIN THẬT:
   - Evidence CHỦ ĐỘNG XÁC NHẬN claim → TRUE
   - Evidence CUNG CẤP CÙNG THÔNG TIN với claim → TRUE
   - ⚠️ VỚI TIN THẬT: Nếu CHỦ ĐỀ + SỰ KIỆN CHÍNH khớp → TRUE với confidence 0.6-0.7
     (không cần mọi chi tiết phụ đều khớp)
   - ⚠️ "Im lặng" (evidence không đề cập chi tiết phụ) ≠ FALSE → có thể TRUE

2. FALSE (SAI) - CHỈ KHI CÓ MÂU THUẪN TRỰC TIẾP:
   - Evidence MÂU THUẪN TRỰC TIẾP với claim (cùng sự kiện, thông tin khác nhau rõ ràng)
   - Evidence NÓI RÕ RÀNG rằng claim SAI
   - ⚠️ CRITICAL: "Không tìm thấy thông tin" ≠ FALSE! Chỉ FALSE khi CÓ BẰNG CHỨNG NGƯỢC LẠI
   - ⚠️ CRITICAL: ZERO TOLERANCE CHO SỐ LIỆU - MỌI CON SỐ ĐỀU PHẢI KHỚP CHÍNH XÁC:
     * Claim: "66 trận đấu, 40 bàn" → Evidence phải nói CHÍNH XÁC "66" và "40"
       Nếu Evidence nói "hơn 40 bàn", "46 bàn", hoặc bất kỳ số nào KHÁC → FALSE
     * Claim: "370.000 USD" → Evidence phải nói CHÍNH XÁC "370000" hoặc "370.000"
       Nếu không khớp chính xác → FALSE
     * Claim: "80%" → Evidence phải nói CHÍNH XÁC "80%"
       "Khoảng 80%" ≠ "80%" chính xác → FALSE
     * SAI ĐƠN VỊ = MÂU THUẪN: "1.000 tỷ" vs "500 tỷ" → FALSE
   - ⚠️ CRITICAL: Nếu claim đưa ra CON SỐ MÂU THUẪN VỚI CHÍNH NÓ → FALSE
   - ⚠️ CRITICAL: Mâu thuẫn NGÀY THÁNG cụ thể (VD: claim nói 23/3, evidence nói 26/4) → FALSE
     Ngày tháng là THÔNG TIN CHÍNH, không phải chi tiết thứ yếu.
   - ⚠️ CRITICAL: Một tuyên bố có NHIỀU CLAIM con — nếu BẤT KỲ claim con nào là FALSE,
     thì tuyên bố tổng thể là FALSE.
   - ⚠️ CRITICAL: Sai lệch CHỨC DANH hoặc TÊN nhân vật chính (VD: "Phó Phát ngôn" vs "Người Phát ngôn")
     → FALSE. Không được coi là chi tiết phụ!
   - ⚠️ CRITICAL: Sai lệch CỤM TỪ KỸ THUẬT NGOẠI GIAO. Claim nói "Đối tác Chiến lược" nhưng
     evidence dùng cụm từ thấp hơn ("tầm cao mới", "Đối tác Toàn diện", "hợp tác sâu rộng", "nâng cấp lên mức cao hơn")
     → FALSE. Không được dùng suy diễn để coi "tầm cao mới" = "Chiến lược".

3. UNCERTAIN (CHƯA RÕ) - Khi:
   - Evidence không liên quan đến claim
   - Không đủ thông tin để kết luận
   - Evidence mâu thuẫn CHÍNH NÓ
   - ⚠️ KHÔNG dùng UNCERTAIN cho trường hợp "tin thật đúng chủ đề nhưng thiếu chi tiết phụ"
     → ĐÂY LÀ TRUE, không phải UNCERTAIN
   - ⚠️ CHỈ dùng UNCERTAIN khi chi tiết bị thiếu là chi tiết PHỤ thực sự
     (VD: số điện thoại, địa chỉ phụ, người tham dự phụ)
     KHÔNG dùng cho: ngày tháng, tên người, chức danh, cụm từ kỹ thuật ngoại giao, con số cốt lõi

VÍ DỤ (QUAN TRỌNG):
- Claim: "Ngày 23/3, ông A phát biểu về vấn đề X"
  Evidence: "Ngày 23/3, ông A phát biểu" (không nói về X) → TRUE (chủ đề chính đúng, chi tiết phụ im lặng)
- Claim: "Ngày 23/3, ông A phát biểu..."
  Evidence: "Ngày 22/3, ông A phát biểu..." → FALSE (mâu thuẫn ngày - ngày là thông tin chính)
- Claim: "Số tiền là 1.000 tỷ đồng"
  Evidence: "Dự án được phân bổ vốn đầu tư công" → TRUE (im lặng ≠ mâu thuẫn)
- Claim: "Số tiền là 1.000 tỷ đồng"
  Evidence: "Dự án được phân bổ 500 tỷ đồng" → FALSE (mâu thuẫn số cụ thể)
- Claim: "Báo Nông thôn ngày nay đã có công văn gửi cơ quan tỉnh Hòa Bình"
  Evidence: "Cục Báo chí gửi công văn" → TRUE (tin thật, chủ đề đúng)
- Claim: "Tin rắc: ông X nói Y vào ngày Z"
  Evidence: "Ông X nói Y vào ngày Z" → TRUE (xác nhận chủ đề)

Trả về JSON: {{"verdict": "true|false|uncertain", "confidence": 0.0-1.0, "reasoning": "..."}}

⚠️ REASONING: TỐI ĐA 3 CÂU, tập trung vào:
- Cụm từ kỹ thuật ngoại giao (nếu có): "Đối tác Chiến lược", "Toàn diện", "tầm cao mới"
- Con số cốt lõi (nếu có): số tiền, số nạn nhân, ngày tháng
- Kết luận: Khớp/Mâu thuẫn/Thiếu thông tin

LƯU Ý QUAN TRỌNG về độ nhạy:
- Phân biệt CHỦ ĐỀ (topic) với CHI TIẾT (details). Sai chủ đề → FALSE. Sai chi tiết → TRUE (tin thật).
- ⚠️ TIN THẬT: Đúng chủ đề + sự kiện chính → TRUE với confidence 0.6-0.7
- ⚠️ TIN RẮC: Đúng chủ đề nhưng sai chi tiết → TRUE chứ không phải FALSE
- ⚠️ FAKE: CHỈ khi có BẰNG CHỨNG MÂU THUẪN TRỰC TIẾP (khác số, khác ngày, sai tên/chức danh chính)
- ⚠️ KHÔNG BAO GIỜ: Đánh FAKE chỉ vì "không tìm thấy" hoặc "im lặng"
- ⚠️ CẢNH BÁO: Không được coi các cụm từ chung chung (VD: "tầm cao mới", "hợp tác sâu rộng", "nâng cấp lên mức cao hơn")
  là bằng chứng xác nhận cho các cụm từ kỹ thuật cụ thể trong claim
  (VD: "Quan hệ Chiến lược", "Đối tác Chiến lược Toàn diện", "nâng cấp thành...").
  Evidence phải dùng ĐÚNG cụm từ kỹ thuật trong claim mới → TRUE.
  Nếu evidence dùng cụm từ ở mức độ thấp hơn
  (VD: claim nói "Chiến lược" nhưng evidence nói "Toàn diện") → FALSE.
  Khác cụm từ kỹ thuật = Mâu thuẫn trực tiếp.
- ⚠️ CẢNH BÁO: Nếu claim nói "Chi tiết X đã được công bố/xác nhận"
  nhưng evidence chỉ nói "đang xem xét/dự kiến/trao đổi" → FALSE.
- ⚠️ CẢNH BÁO: Tên người phát ngôn hoặc chức danh sai lệch nhẹ
  (VD: Phó phát ngôn vs Người phát ngôn) → FALSE (nếu là nhân vật chính).
- ⚠️ CẢNH BÁO: "Sự kiện tương tự nhưng khác thời gian/nội dung"
  = FALSE, không phải TRUE."""

    prompt = f"""KIỂM CHỨNG TUYÊN BỐ:

CLAIM: {claim}

EVIDENCE:
{evidence_text}

PHÂN TÍCH TỪNG BƯỚC:
1. Xác định CHỦ ĐỀ chính của claim (sự kiện gì?)
2. KIỂM TRA SỐ LIỆU CHÍNH XÁC: Trích xuất MỌI con số trong claim
3. ĐỐI CHIẾU từng con số với evidence - phải khớp CHÍNH XÁC
4. Tìm evidence MÂU THUẨN TRỰC TIẾP với CHỦ ĐỀ không?
5. Tìm evidence XÁC NHẬN CHỦ ĐỘNG chủ đề không?

QUYẾT ĐỊNH:
- Con số trong claim mà evidence đưa ra số KHÁC → FALSE (ZERO TOLERANCE)
- Số trong claim không đề cập trong evidence → TRUE (im lặng ≠ mâu thuẫn)
- Evidence chủ động xác nhận + số khớp → TRUE
- Không xác định được → UNCERTAIN

Trả về CHỈ JSON (reasoning tối đa 3 câu):"""

    try:
        response = await model.ainvoke(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        )

        content = response.content if hasattr(response, "content") else str(response)
        logger.info(f"[AGENT] LLM response length: {len(content)}")

        # Parse JSON response
        content = content.strip()
        content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            import re

            match = re.search(r"\{[^}]+\}", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                parsed = {"verdict": "uncertain", "confidence": 0.3}

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

        logger.info(f"[AGENT] Verdict: {verdict} (confidence: {confidence:.2f})")
        logger.info(f"[AGENT] Reasoning: {reasoning[:200]}...")

        return {"verdict": verdict, "confidence": confidence, "label": verdict, "reasoning": reasoning}

    except Exception as e:
        logger.error(f"[AGENT] Verification failed: {e}")
        return {"verdict": "uncertain", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}


def run_verifier_agent_sync(claim: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Synchronous wrapper for run_verifier_agent.

    Handles nested asyncio loops properly (works from within asyncio.run).
    """
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_verifier_agent(claim, evidence))

    def _run_in_thread():
        return asyncio.run(run_verifier_agent(claim, evidence))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_in_thread).result()
