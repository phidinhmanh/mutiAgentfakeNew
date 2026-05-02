"""
Claim Extractor Agent - Direct LLM approach for reliability.

Extracts factual claims from text using direct LLM calls.
Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import json
import logging

from dotenv import load_dotenv

from trust_agents.llm.factory import create_chat_model

load_dotenv()
logger = logging.getLogger("ClaimExtractor.Agent")


async def run_claim_extractor_agent(text: str) -> list[str]:
    """Extract claims from text using direct LLM call."""
    model = create_chat_model()

    logger.info("[AGENT] Claim Extraction Agent (Direct LLM) initialized")

    # Vietnamese prompt for claim extraction
    # Note: prompt lines are intentionally longer than 120 chars for readability
    # ruff: noqa: E501
    system_prompt = """Bạn là chuyên gia trích xuất thông tin thực tế (fact extraction).
Nhiệm vụ: Trích xuất các tuyên bố có thể kiểm chứng từ văn bản tiếng Việt.
QUAN TRỌNG: Giữ nguyên ngôn ngữ gốc (tiếng Việt), KHÔNG dịch.

NGUYÊN TẮC TRÍCH XUẤT (QUAN TRỌNG):
1. GIỮ NGUYÊN THÔNG TIN COMPOUND: Nếu một tuyên bố chứa nhiều thông tin (VD: "3 tuần làm việc thám hiểm hang động"), giữ nó LÀM MỘT tuyên bố duy nhất. KHÔNG tách thành "3 tuần", "làm việc", "thám hiểm" riêng biệt.
2. KHÔNG tạo claim mới từ kết hợp/ghép các phần của thông tin (VD: không tạo "3 ngày 2 đêm" nếu văn bản chỉ nói "3 tuần").
3. SỐ LIỆU CÙNG MỘT CHỦ ĐỀ: "Doanh thu 500 triệu USD" + "thời gian 6 tháng" → giữ chung MỘT claim ("Doanh thu 500 triệu USD trong 6 tháng").
4. CHỈ tách claim khi các thông tin ĐỘC LẬP VỀ CHỦ ĐỀ (VD: "Họp báo tại Hà Nội" + "Chủ đề về kinh tế" → có thể tách).
5. Ưu tiên tối thiểu hóa số lượng claims. Tốt nhất: 1-3 claims.

CẤU TRÚC S-V-O (QUAN TRỌNG CHO TASK 1):
- Mỗi claim phải là một CÂU ĐỘC LẬP, chứa đầy đủ:
  * CHỦ NGỮ cụ thể (TÊN RIÊNG: người, tổ chức, địa điểm - KHÔNG dùng đại từ)
  * VỊ NGỮ rõ ràng (hành động/sự kiện)
  * ĐỐI TƯỢNG cụ thể (thông tin cần kiểm chứng)
- KHÔNG dùng đại từ: "ông ấy", "bà đó", "nhóm nhạc đó", "công ty này" → phải thay bằng tên cụ thể từ văn bản gốc
- Mỗi claim phải TỰ GIẢI THÍCH được nếu đứng một mình (không cần ngữ cảnh thêm)

TRẢ VỀ JSON: {"claims": ["tuyên bố 1", "tuyên bố 2"]}"""

    # ruff: noqa: E501
    prompt = f"""Trích xuất các tuyên bố thực tế từ văn bản sau.

NGUYÊN TẮC:
- Mỗi claim = MỘT chủ đề hoặc MỘT sự kiện cụ thể
- GIỮ NGUYÊN cấu trúc gốc: "3 tuần làm việc" = 1 claim (KHÔNG tách thành "3" + "tuần")
- KHÔNG ghép/tạo thông tin mới không có trong văn bản gốc
- Nếu thông tin có số + thời gian + địa điểm cùng 1 chủ đề → giữ làm 1 claim
- Mỗi claim phải có NGHĨA ĐẦY ĐỦ nếu đứng một mình
- CẤU TRÚC S-V-O: Mỗi claim phải có CHỦ NGỮ CỤ THỂ + VỊ NGỮ + ĐỐI TƯỢNG
- Thay thế mọi đại từ ("ông ấy", "bà đó", "nhóm nhạc đó") bằng tên cụ thể từ văn bản

Văn bản: {text}

Trả về CHỈ JSON, không có giải thích:"""

    try:
        response = await model.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])

        content = response.content if hasattr(response, "content") else str(response)
        logger.info(f"[AGENT] LLM response length: {len(content)}")

        # Parse JSON response
        content = content.strip()
        content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                parsed = {"claims": []}

        claims = parsed.get("claims", [])
        if isinstance(claims, list):
            claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]

        if claims:
            logger.info(f"[AGENT] Successfully extracted {len(claims)} claims")
            return claims

        # Fallback: return original text as single claim if extraction fails
        if text.strip():
            logger.warning("[AGENT] LLM extraction failed, using text as single claim")
            return [text.strip()]

        return []

    except Exception as e:
        logger.error(f"[AGENT] Claim extraction failed: {e}")
        # Fallback: return original text
        if text.strip():
            return [text.strip()]
        return []


def run_claim_extractor_agent_sync(text: str) -> list[str]:
    """Synchronous wrapper for run_claim_extractor_agent.

    Handles nested asyncio loops properly (works from within asyncio.run).
    """
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_claim_extractor_agent(text))

    def _run_in_thread():
        return asyncio.run(run_claim_extractor_agent(text))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_in_thread).result()
