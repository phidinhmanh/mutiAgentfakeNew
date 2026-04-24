"""Streamlit UI for Vietnamese Fake News Detection."""
import logging
import time
from typing import Any

import streamlit as st

from fake_news_detector.application.analysis_service import (
    analyze_with_legacy,
    analyze_with_trust,
)
from fake_news_detector.application.index_service import (
    build_vector_index,
    load_sample_claim_and_evidence,
)
from fake_news_detector.config import settings
from fake_news_detector.ui.session import init_session_state
from fake_news_detector.visualization.wordcloud import (
    analyze_text_length,
    get_top_words,
)

try:
    from trust_agents.orchestrator import TRUSTOrchestrator

    TRUST_AVAILABLE = True
except ImportError as e:
    TRUST_AVAILABLE = False
    logging.warning(f"TRUST orchestrator not available: {e}")

try:
    from fake_news_detector.agents.claim_extractor import (
        extract_claims,
        filter_verifiable_claims,
    )
    from fake_news_detector.agents.evidence_retriever import (
        enrich_evidence_with_context,
        retrieve_evidence_for_claims,
    )

    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_article(article: str, use_trust: bool = True) -> dict[str, Any]:
    """Run full analysis pipeline on article."""
    start_time = time.time()

    if use_trust and TRUST_AVAILABLE:
        logger.info("Using TRUST multi-agent system")
        try:
            results = analyze_with_trust(
                article=article,
                orchestrator=TRUSTOrchestrator(top_k_evidence=5),
            )
            logger.info(f"TRUST pipeline complete: {time.time() - start_time:.2f}s")
            return results
        except Exception as e:
            logger.error(f"TRUST pipeline failed: {e}")
            fallback_results = analyze_with_legacy(
                article=article,
                extract_claims_fn=lambda _article: [],
                filter_verifiable_claims_fn=lambda claims: claims,
                retrieve_evidence_for_claims_fn=lambda claims: [],
                enrich_evidence_with_context_fn=lambda evidence, text: evidence,
            )
            fallback_results["trust_error"] = str(e)
            fallback_results["claims"] = []
            return fallback_results

    logger.info("Using legacy single-agent system")
    if LEGACY_AVAILABLE:
        results = analyze_with_legacy(
            article=article,
            extract_claims_fn=extract_claims,
            filter_verifiable_claims_fn=filter_verifiable_claims,
            retrieve_evidence_for_claims_fn=retrieve_evidence_for_claims,
            enrich_evidence_with_context_fn=enrich_evidence_with_context,
        )
        logger.info(f"Analysis complete: {time.time() - start_time:.2f}s")
        return results

    results = analyze_with_legacy(
        article=article,
        extract_claims_fn=lambda _article: [],
        filter_verifiable_claims_fn=lambda claims: claims,
        retrieve_evidence_for_claims_fn=lambda claims: [],
        enrich_evidence_with_context_fn=lambda evidence, text: evidence,
    )
    logger.info(f"Analysis complete: {time.time() - start_time:.2f}s")
    return results


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Vietnamese Fake News Detector",
        page_icon="🔍",
        layout="wide",
    )

    init_session_state()

    st.title("🔍 Vietnamese Fake News Detection")
    st.markdown("Multi-Agent RAG System với **TRUST Agents** + Gemini/LLaMA")

    with st.sidebar:
        st.header("Cài đặt")

        st.subheader("Pipeline")
        pipeline_options = ["trust"] if TRUST_AVAILABLE else []
        if LEGACY_AVAILABLE:
            pipeline_options.append("legacy")
        if not pipeline_options:
            pipeline_options = ["none"]

        selected_pipeline = st.radio(
            "Chọn pipeline:",
            options=pipeline_options,
            format_func=lambda value: {
                "trust": "🤖 TRUST Multi-Agent (Mới)",
                "legacy": "📝 Single-Agent (Cũ)",
                "none": "⚠️ Không có pipeline",
            }.get(value, value),
            index=0,
        )
        st.session_state["pipeline"] = selected_pipeline

        if selected_pipeline == "trust" and not TRUST_AVAILABLE:
            st.error("TRUST Agents không khả dụng. Kiểm tra cài đặt.")

        st.divider()

        st.subheader("Tải Dataset mẫu")
        if st.button("Load ViFactCheck Sample"):
            with st.spinner("Đang tải..."):
                try:
                    sample_claim, sample_evidence = load_sample_claim_and_evidence()
                    st.session_state["sample_claim"] = sample_claim
                    st.session_state["sample_evidence"] = sample_evidence
                    st.success("Đã tải mẫu!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        st.subheader("Tải FAISS Index")
        if st.button("Build Vector Index"):
            with st.spinner("Đang tạo index..."):
                try:
                    build_vector_index()
                    st.success("Index đã được tạo!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        st.subheader("Mô hình")
        if TRUST_AVAILABLE:
            st.success("✅ TRUST Agents: Hoạt động")
        else:
            st.warning("⚠️ TRUST Agents: Không khả dụng")
        st.write(f"PhoBERT: {settings.phobert_model}")
        st.write(f"Embedding: {settings.embedding_model}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Nội dung bài viết")
        article = st.text_area(
            "Nhập nội dung tin tức:",
            height=400,
            placeholder="Dán nội dung bài viết cần kiểm tra...",
        )

        col_analyze, col_clear = st.columns(2)
        with col_analyze:
            analyze_button = st.button(
                "🔍 Phân tích", type="primary", use_container_width=True
            )
        with col_clear:
            if st.button("🗑️ Xóa", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    with col2:
        st.header("📊 Kết quả")

        if analyze_button and article:
            use_trust = st.session_state["pipeline"] == "trust"
            with st.spinner("Đang phân tích..."):
                st.session_state["analysis_results"] = analyze_article(
                    article, use_trust=use_trust
                )

        if st.session_state.get("analysis_results"):
            results = st.session_state["analysis_results"]

            if "baseline" in results:
                baseline = results["baseline"]
                col_real, col_fake, col_conf = st.columns(3)
                with col_real:
                    st.metric("REAL", f"{baseline.get('real_prob', 0):.1%}")
                with col_fake:
                    st.metric("FAKE", f"{baseline.get('fake_prob', 0):.1%}")
                with col_conf:
                    st.metric("Confidence", f"{baseline.get('confidence', 0):.1%}")

                verdict_color = (
                    "red" if baseline.get("fake_prob", 0) > 0.5 else "green"
                )
                baseline_label = (
                    "FAKE" if baseline.get("fake_prob", 0) > 0.5 else "REAL"
                )
                st.markdown(
                    f"**Baseline Verdict:** :{verdict_color}[{baseline_label}]"
                )

            if "trust" in results:
                st.divider()
                st.header("🤖 TRUST Agents Results")
                trust_summary = results["trust"].get("summary", {})
                verdicts = results.get("verdicts", [])

                summary_data = trust_summary.get("verdicts", {})
                col_true, col_false, col_uncertain = st.columns(3)
                with col_true:
                    st.metric("✅ TRUE", summary_data.get("true", 0))
                with col_false:
                    st.metric("❌ FALSE", summary_data.get("false", 0))
                with col_uncertain:
                    st.metric("❓ UNCERTAIN", summary_data.get("uncertain", 0))

                avg_conf = trust_summary.get("average_confidence", 0)
                st.progress(avg_conf, text=f"Average Confidence: {avg_conf:.1%}")

                if verdicts:
                    st.subheader("📋 Chi tiết Claims")
                    for index, verdict_info in enumerate(verdicts):
                        verdict = verdict_info.get("verdict", "uncertain")
                        confidence = verdict_info.get("confidence", 0.5)
                        if verdict == "true":
                            verdict_emoji = "✅"
                            verdict_color = "green"
                        elif verdict == "false":
                            verdict_emoji = "❌"
                            verdict_color = "red"
                        else:
                            verdict_emoji = "❓"
                            verdict_color = "gray"

                        claim_text = verdict_info.get("claim", "")[:100]
                        with st.expander(
                            f"{verdict_emoji} Claim {index + 1}: {claim_text}..."
                        ):
                            st.write(
                                f"**Verdict:** :{verdict_color}[{verdict.upper()}]"
                            )
                            st.write(f"**Confidence:** {confidence:.1%}")
                            st.write(
                                f"**Reasoning:** {verdict_info.get('reasoning', 'N/A')}"
                            )

            if "trust_error" in results:
                st.error(f"TRUST Pipeline Error: {results['trust_error']}")

            if "claims_with_evidence" in results:
                st.divider()
                st.header("📋 Claims (Legacy)")
                for index, claim_with_evidence in enumerate(
                    results["claims_with_evidence"][:5]
                ):
                    title = claim_with_evidence.get("text", "")[:60]
                    with st.expander(f"Claim {index + 1}: {title}..."):
                        st.write(
                            f"**Type:** {claim_with_evidence.get('type', 'UNKNOWN')}"
                        )
                        st.write(
                            "**Verifiable:** "
                            f"{claim_with_evidence.get('verifiable', False)}"
                        )
                        st.write(f"**Text:** {claim_with_evidence.get('text', '')}")

            if "stylistic_features" in results:
                st.divider()
                st.subheader("Đặc điểm văn phong")
                features = results["stylistic_features"]
                col1_f, col2_f = st.columns(2)
                with col1_f:
                    st.write(f"- Tỷ lệ HOA: {features.get('caps_ratio', 0):.2%}")
                    st.write(f"- Cảm xúc: {features.get('emotional_markers', 0)}")
                    st.write(f"- Giật gân: {features.get('sensational_words', 0)}")
                with col2_f:
                    st.write(f"- Nguồn: {features.get('source_mentions', 0)}")
                    st.write(
                        f"- Câu TB: {features.get('avg_sentence_length', 0):.1f}"
                    )
                    st.write(f"- Fake Score: {features.get('fake_score', 0):.2f}")

            if "evidence" in results and results["evidence"]:
                st.divider()
                st.header("📚 Evidence Retrieved")
                for index, evidence in enumerate(results["evidence"][:5]):
                    title = evidence.get("title", "")[:60]
                    with st.expander(f"Evidence {index + 1}: {title}..."):
                        st.write(f"**Source:** {evidence.get('source', 'unknown')}")
                        st.write(f"**Score:** {evidence.get('score', 0):.3f}")
                        st.write(f"**Content:** {evidence.get('content', '')}")

            with st.expander("📈 Word Analysis"):
                if article:
                    stats = analyze_text_length(article)
                    st.write(
                        f"Từ: {stats['word_count']}, Câu: {stats['sentence_count']}"
                    )
                    top_words = get_top_words(article, 10)
                    st.write("Top words:", ", ".join(word for word, _ in top_words))


if __name__ == "__main__":
    main()
