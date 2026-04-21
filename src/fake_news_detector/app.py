"""Streamlit UI for Vietnamese Fake News Detection."""
import json
import logging
import time
from typing import Any, Generator

import streamlit as st
from openai import OpenAI

from fake_news_detector.config import settings
from fake_news_detector.data.loader import load_vifactcheck
from fake_news_detector.data.preprocessing import summarize_for_long_text
from fake_news_detector.models.baseline import get_baseline_model
from fake_news_detector.models.stylistic import extract_stylistic_features
from fake_news_detector.rag.retriever import get_vector_store
from fake_news_detector.visualization.wordcloud import (
    analyze_text_length,
    get_top_words,
)

# Try to import TRUST orchestrator (new multi-agent system)
try:
    from trust_agents.orchestrator import TRUSTOrchestrator, fact_check
    TRUST_AVAILABLE = True
except ImportError as e:
    TRUST_AVAILABLE = False
    logging.warning(f"TRUST orchestrator not available: {e}")

# Legacy imports (for backward compatibility)
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


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    defaults = {
        "analysis_results": None,
        "claims": [],
        "selected_claims": [],
        "evidence": [],
        "verdicts": [],
        "baseline_result": None,
        "stylistic_features": None,
        "use_trust_agents": True,  # Default to use TRUST agents
        "pipeline": "trust",  # 'trust' or 'legacy'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def analyze_article(article: str, use_trust: bool = True) -> dict[str, Any]:
    """Run full analysis pipeline on article.

    Args:
        article: Input article text
        use_trust: Whether to use TRUST multi-agent system (default: True)

    Returns:
        Complete analysis results
    """
    start_time = time.time()
    results: dict[str, Any] = {}

    if len(article) > 3000:
        article = summarize_for_long_text(article, max_chars=3000)
        results["summarized"] = True

    # Always run baseline model
    baseline_model = get_baseline_model()
    baseline_result = baseline_model.predict_with_sliding_window(article)
    results["baseline"] = baseline_result
    logger.info(f"Baseline done: {time.time() - start_time:.2f}s")

    if use_trust and TRUST_AVAILABLE:
        # Use TRUST multi-agent system
        logger.info("Using TRUST multi-agent system")
        try:
            orchestrator = TRUSTOrchestrator(top_k_evidence=5)
            trust_result = orchestrator.process_text(article)

            results["trust"] = {
                "claims": trust_result.claims,
                "results": trust_result.results,
                "summary": trust_result.summary,
            }
            results["claims"] = [{"text": c, "source": "trust"} for c in trust_result.claims]

            # Extract verdicts from TRUST results
            verdicts = []
            for r in trust_result.results:
                verdicts.append({
                    "verdict": r.get("verdict", "uncertain"),
                    "confidence": r.get("confidence", 0.3),
                    "label": r.get("label", "uncertain"),
                    "reasoning": r.get("reasoning", ""),
                    "claim": r.get("claim", ""),
                })
            results["verdicts"] = verdicts

            logger.info(f"TRUST pipeline complete: {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"TRUST pipeline failed: {e}")
            results["trust_error"] = str(e)
            # Fall back to legacy
            results["claims"] = []
    else:
        # Use legacy single-agent approach
        logger.info("Using legacy single-agent system")
        if LEGACY_AVAILABLE:
            claims = extract_claims(article)
            results["claims"] = claims
            logger.info(f"Extracted {len(claims)} claims")

            verifiable_claims = filter_verifiable_claims(claims)
            results["verifiable_claims"] = verifiable_claims
            logger.info(f"Found {len(verifiable_claims)} verifiable claims")

            if verifiable_claims:
                claims_with_evidence = retrieve_evidence_for_claims(verifiable_claims)

                for cw in claims_with_evidence:
                    cw["evidence"] = enrich_evidence_with_context(
                        cw.get("evidence", []),
                        cw.get("text", ""),
                    )

                results["claims_with_evidence"] = claims_with_evidence

                merged_evidence = []
                for cw in claims_with_evidence:
                    merged_evidence.extend(cw.get("evidence", []))
                results["evidence"] = merged_evidence[:10]

    results["stylistic_features"] = extract_stylistic_features(article)
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

        # Pipeline selection
        st.subheader("Pipeline")
        pipeline_options = ["trust"] if TRUST_AVAILABLE else []
        if LEGACY_AVAILABLE:
            pipeline_options.append("legacy")

        if not pipeline_options:
            pipeline_options = ["none"]

        selected_pipeline = st.radio(
            "Chọn pipeline:",
            options=pipeline_options,
            format_func=lambda x: {
                "trust": "🤖 TRUST Multi-Agent (Mới)",
                "legacy": "📝 Single-Agent (Cũ)",
                "none": "⚠️ Không có pipeline"
            }.get(x, x),
            index=0 if "trust" in pipeline_options else 0,
        )
        st.session_state["pipeline"] = selected_pipeline

        if selected_pipeline == "trust" and not TRUST_AVAILABLE:
            st.error("TRUST Agents không khả dụng. Kiểm tra cài đặt.")

        st.divider()

        st.subheader("Tải Dataset mẫu")
        if st.button("Load ViFactCheck Sample"):
            with st.spinner("Đang tải..."):
                try:
                    dataset = load_vifactcheck("train")
                    sample = dataset[0]
                    st.session_state["sample_claim"] = sample.get("claim", "")
                    st.session_state["sample_evidence"] = sample.get("evidence", "")
                    st.success("Đã tải mẫu!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        st.subheader("Tải FAISS Index")
        if st.button("Build Vector Index"):
            with st.spinner("Đang tạo index..."):
                try:
                    dataset = load_vifactcheck("train")
                    vector_store = get_vector_store()
                    docs = [
                        {"content": item.get("evidence", ""), "label": item.get("label", "")}
                        for item in dataset
                        if item.get("evidence")
                    ]
                    vector_store.add_documents(docs[:1000])
                    vector_store.save(settings.faiss_index_path)
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
            analyze_button = st.button("🔍 Phân tích", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Xóa", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    with col2:
        st.header("📊 Kết quả")

        if analyze_button and article:
            use_trust = st.session_state["pipeline"] == "trust"
            with st.spinner("Đang phân tích..."):
                results = analyze_article(article, use_trust=use_trust)
                st.session_state["analysis_results"] = results

        if st.session_state.get("analysis_results"):
            results = st.session_state["analysis_results"]

            # Baseline results (always shown)
            if "baseline" in results:
                baseline = results["baseline"]
                col_real, col_fake, col_conf = st.columns(3)

                with col_real:
                    st.metric(
                        "REAL",
                        f"{baseline.get('real_prob', 0):.1%}",
                        delta=None,
                    )
                with col_fake:
                    st.metric(
                        "FAKE",
                        f"{baseline.get('fake_prob', 0):.1%}",
                        delta=None,
                    )
                with col_conf:
                    st.metric(
                        "Confidence",
                        f"{baseline.get('confidence', 0):.1%}",
                        delta=None,
                    )

                verdict_color = "red" if baseline.get("fake_prob", 0) > 0.5 else "green"
                st.markdown(
                    f"**Baseline Verdict:** :{verdict_color}[{'FAKE' if baseline.get('fake_prob', 0) > 0.5 else 'REAL'}]"
                )

            # TRUST Agents results
            if "trust" in results:
                st.divider()
                st.header("🤖 TRUST Agents Results")

                trust_summary = results["trust"].get("summary", {})
                verdicts = results.get("verdicts", [])

                # Summary stats
                summary_data = trust_summary.get("verdicts", {})
                col_true, col_false, col_uncertain = st.columns(3)
                with col_true:
                    st.metric("✅ TRUE", summary_data.get("true", 0))
                with col_false:
                    st.metric("❌ FALSE", summary_data.get("false", 0))
                with col_uncertain:
                    st.metric("❓ UNCERTAIN", summary_data.get("uncertain", 0))

                # Average confidence
                avg_conf = trust_summary.get("average_confidence", 0)
                st.progress(avg_conf, text=f"Average Confidence: {avg_conf:.1%}")

                # Individual claim results
                if verdicts:
                    st.subheader("📋 Chi tiết Claims")
                    for i, v in enumerate(verdicts):
                        verdict = v.get("verdict", "uncertain")
                        confidence = v.get("confidence", 0.5)

                        # Color based on verdict
                        if verdict == "true":
                            verdict_emoji = "✅"
                            verdict_color = "green"
                        elif verdict == "false":
                            verdict_emoji = "❌"
                            verdict_color = "red"
                        else:
                            verdict_emoji = "❓"
                            verdict_color = "gray"

                        claim_text = v.get("claim", "")[:100]
                        with st.expander(f"{verdict_emoji} Claim {i+1}: {claim_text}..."):
                            st.write(f"**Verdict:** :{verdict_color}[{verdict.upper()}]")
                            st.write(f"**Confidence:** {confidence:.1%}")
                            st.write(f"**Reasoning:** {v.get('reasoning', 'N/A')}")

            # TRUST error
            if "trust_error" in results:
                st.error(f"TRUST Pipeline Error: {results['trust_error']}")

            # Legacy results
            if "claims_with_evidence" in results:
                st.divider()
                st.header("📋 Claims (Legacy)")

                claims_with_evidence = results["claims_with_evidence"]
                for i, cw in enumerate(claims_with_evidence[:5]):
                    with st.expander(f"Claim {i+1}: {cw.get('text', '')[:60]}..."):
                        st.write(f"**Type:** {cw.get('type', 'UNKNOWN')}")
                        st.write(f"**Verifiable:** {cw.get('verifiable', False)}")
                        st.write(f"**Text:** {cw.get('text', '')}")

            # Stylistic features
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
                    st.write(f"- Câu TB: {features.get('avg_sentence_length', 0):.1f}")
                    st.write(f"- Fake Score: {features.get('fake_score', 0):.2f}")

            # Evidence (from legacy pipeline)
            if "evidence" in results and results["evidence"]:
                st.divider()
                st.header("📚 Evidence Retrieved")
                for i, ev in enumerate(results["evidence"][:5]):
                    with st.expander(f"Evidence {i+1}: {ev.get('title', '')[:60]}..."):
                        st.write(f"**Source:** {ev.get('source', 'unknown')}")
                        st.write(f"**Score:** {ev.get('score', 0):.3f}")
                        st.write(f"**Content:** {ev.get('content', '')}")

            # Word analysis
            with st.expander("📈 Word Analysis"):
                if article:
                    stats = analyze_text_length(article)
                    st.write(f"Từ: {stats['word_count']}, Câu: {stats['sentence_count']}")
                    top_words = get_top_words(article, 10)
                    st.write("Top words:", ", ".join(w[0] for w in top_words))

if __name__ == "__main__":
    main()