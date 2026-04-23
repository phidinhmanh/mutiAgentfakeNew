"""Interactive terminal runner for the fake news detection project."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.config import settings
from fake_news_detector.data.preprocessing import summarize_for_long_text
from fake_news_detector.models.baseline import get_baseline_model
from fake_news_detector.models.stylistic import extract_stylistic_features
from trust_agents.agents.claim_extractor import run_claim_extractor_agent_sync
from trust_agents.agents.evidence_retrieval import run_evidence_retrieval_agent_sync
from trust_agents.agents.explainer import run_explainer_agent_sync
from trust_agents.agents.verifier import run_verifier_agent_sync
from trust_agents.orchestrator import TRUSTOrchestrator

logger = logging.getLogger("interactive_runner")

GOLDEN_SAMPLES = [
    {
        "id": "fake_news_alien",
        "text": "Việt Nam đã phát hiện người ngoài hành tinh tại Hà Nội và chính phủ đang che giấu thông tin này.",
        "expected_verdict": "false",
    },
    {
        "id": "fake_news_gdp_exaggerated",
        "text": "Việt Nam đạt tăng trưởng GDP 50% trong quý 1 năm 2024, cao nhất thế giới.",
        "expected_verdict": "false",
    },
    {
        "id": "real_news_gdp_wb",
        "text": "Theo báo cáo của World Bank, Việt Nam đạt tăng trưởng GDP 5.6% trong năm 2023, thuộc nhóm cao nhất ASEAN.",
        "expected_verdict": "true",
    },
    {
        "id": "unverifiable_generic",
        "text": "Một nguồn tin cho biết lãnh đạo đất nước sẽ có cuộc họp quan trọng vào tuần tới nhưng không tiết lộ chi tiết.",
        "expected_verdict": "uncertain",
    },
]


def configure_logging(verbose: bool) -> None:
    """Configure logging for the interactive runner."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")


def print_header(title: str) -> None:
    """Print a section header."""
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def prompt(message: str, default: str | None = None) -> str:
    """Prompt for a single-line input."""
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{message}{suffix}: ").strip()
    if value:
        return value
    return default or ""


def prompt_yes_no(message: str, default: bool = True) -> bool:
    """Prompt for a yes/no decision."""
    default_label = "Y/n" if default else "y/N"
    value = input(f"{message} [{default_label}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def prompt_int(message: str, default: int) -> int:
    """Prompt for an integer."""
    raw_value = prompt(message, str(default))
    try:
        return int(raw_value)
    except ValueError:
        print(f"Giá trị không hợp lệ, dùng mặc định {default}.")
        return default


def prompt_multiline_text() -> str:
    """Prompt for multi-line text terminated by EOF."""
    print("Nhập nội dung. Gõ 'EOF' trên một dòng riêng để kết thúc.")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "EOF":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def truncate(text: str, limit: int = 160) -> str:
    """Truncate long text for display."""
    value = text.replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def print_json(data: Any) -> None:
    """Pretty-print JSON data."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def startup_checks() -> None:
    """Print basic runtime information."""
    print_header("Interactive Fake News Runner")
    print(f"Project root      : {ROOT_DIR}")
    print(f"FAISS index path  : {Path(settings.faiss_index_path).resolve()}")
    print(f"LLM model         : {settings.llm_model}")
    print(f"Embedding model   : {settings.embedding_model}")
    print(f"PhoBERT model     : {settings.phobert_model}")
    print(f"Search provider   : {settings.search_provider}")
    print(f"HF token present  : {'yes' if bool(settings.hf_token) else 'no'}")
    print(f"SERPER key present: {'yes' if bool(settings.serper_api_key) else 'no'}")
    print(f"TAVILY key present: {'yes' if bool(settings.tavily_api_key) else 'no'}")
    print(f"NVIDIA key present: {'yes' if bool(settings.nvidia_api_key) else 'no'}")


def get_article_input() -> str:
    """Get article input from the user."""
    if prompt_yes_no("Nhập nhiều dòng?", default=True):
        article = prompt_multiline_text()
    else:
        article = prompt("Nhập nội dung")
    return article.strip()


def run_baseline_only(article: str) -> dict[str, Any]:
    """Run the baseline model."""
    baseline_model = get_baseline_model()
    return baseline_model.predict_with_sliding_window(article)


def run_trust_only(article: str, skip_evidence: bool, top_k_evidence: int) -> Any:
    """Run the TRUST pipeline."""
    orchestrator = TRUSTOrchestrator(top_k_evidence=top_k_evidence)
    return orchestrator.process_text(article, skip_evidence=skip_evidence)


def run_full_analysis(article: str, skip_evidence: bool, top_k_evidence: int) -> dict[str, Any]:
    """Run the full article analysis flow."""
    processed_article = article
    summarized = False
    if len(processed_article) > 3000:
        processed_article = summarize_for_long_text(processed_article, max_chars=3000)
        summarized = True

    baseline_result = run_baseline_only(processed_article)
    trust_result = run_trust_only(processed_article, skip_evidence=skip_evidence, top_k_evidence=top_k_evidence)
    stylistic_result = extract_stylistic_features(processed_article)

    return {
        "input_text": article,
        "processed_text": processed_article,
        "summarized": summarized,
        "baseline": baseline_result,
        "trust": {
            "original_text": trust_result.original_text,
            "claims": trust_result.claims,
            "results": trust_result.results,
            "summary": trust_result.summary,
        },
        "stylistic_features": stylistic_result,
    }


def print_baseline_result(result: dict[str, Any]) -> None:
    """Print baseline output."""
    print_header("Baseline Result")
    print(f"Label      : {result.get('label', 'UNKNOWN')}")
    print(f"Confidence : {result.get('confidence', 0.0):.1%}")
    print(f"Fake prob  : {result.get('fake_prob', 0.0):.1%}")
    print(f"Real prob  : {result.get('real_prob', 0.0):.1%}")
    if "num_chunks" in result:
        print(f"Chunks     : {result.get('num_chunks')}")


def print_trust_result(result: dict[str, Any]) -> None:
    """Print TRUST pipeline output."""
    print_header("TRUST Summary")
    summary = result.get("summary", {})
    verdicts = summary.get("verdicts", {})
    print(f"Claims             : {len(result.get('claims', []))}")
    print(f"Total claims       : {summary.get('total_claims', 0)}")
    print(f"Average confidence : {summary.get('average_confidence', 0.0):.1%}")
    print(f"Verdicts           : {verdicts}")

    print_header("TRUST Claim Details")
    for index, item in enumerate(result.get("results", []), start=1):
        print(f"[{index}] Claim      : {item.get('claim', 'N/A')}")
        print(f"    Verdict    : {item.get('verdict', 'uncertain')}")
        print(f"    Confidence : {item.get('confidence', 0.0):.1%}")
        print(f"    Reasoning  : {truncate(item.get('reasoning', 'N/A'), 240)}")
        if item.get("summary"):
            print(f"    Summary    : {truncate(item.get('summary', ''), 240)}")
        if item.get("explanation"):
            print(f"    Explanation: {truncate(item.get('explanation', ''), 240)}")
        print()


def print_stylistic_result(result: dict[str, Any]) -> None:
    """Print stylistic features."""
    print_header("Stylistic Features")
    print(f"Fake score         : {result.get('fake_score', 0.0):.2f}")
    print(f"Sentences          : {result.get('num_sentences', 0)}")
    print(f"Words              : {result.get('num_words', 0)}")
    print(f"Avg sentence length: {result.get('avg_sentence_length', 0.0):.2f}")
    print(f"Caps ratio         : {result.get('caps_ratio', 0.0):.2%}")
    print(f"Number ratio       : {result.get('number_ratio', 0.0):.2%}")
    print(f"Emotional markers  : {result.get('emotional_markers', 0)}")
    print(f"Sensational words  : {result.get('sensational_words', 0)}")
    print(f"Source mentions    : {result.get('source_mentions', 0)}")


def build_documents_from_dataset(limit: int) -> list[dict[str, Any]]:
    """Build vector-store documents from the dataset."""
    from fake_news_detector.data.loader import load_vifactcheck
    dataset = load_vifactcheck("train")
    documents: list[dict[str, Any]] = []
    for item in dataset[:limit]:
        content = item.get("evidence", "")
        if not content:
            continue
        documents.append(
            {
                "content": content,
                "label": item.get("label", ""),
                "claim": item.get("claim", ""),
                "source": item.get("source", "vi_fact_check"),
                "claim_date": item.get("claim_date", ""),
            }
        )
    return documents


def browse_dataset_samples() -> None:
    """Browse samples from ViFactCheck."""
    from fake_news_detector.data.loader import load_vifactcheck
    print_header("Dataset Browser")
    split = prompt("Dataset split", "train")
    dataset = load_vifactcheck(split)
    print(f"Tổng số mẫu: {len(dataset)}")

    while True:
        raw_index = prompt("Nhập index mẫu hoặc 'q' để thoát", "0")
        if raw_index.lower() == "q":
            break

        try:
            index = int(raw_index)
        except ValueError:
            print("Index không hợp lệ.")
            continue

        if index < 0 or index >= len(dataset):
            print("Index ngoài phạm vi.")
            continue

        sample = dataset[index]
        print_header(f"Sample {index}")
        print(f"Claim      : {sample.get('claim', '')}")
        print(f"Evidence   : {truncate(sample.get('evidence', ''), 500)}")
        print(f"Label      : {sample.get('label', '')}")
        print(f"Source     : {sample.get('source', '')}")
        print(f"Claim date : {sample.get('claim_date', '')}")

        if prompt_yes_no("Chạy full analysis với claim này?", default=False):
            analysis = run_full_analysis(sample.get("claim", ""), skip_evidence=False, top_k_evidence=5)
            print_baseline_result(analysis["baseline"])
            print_trust_result(analysis["trust"])
            print_stylistic_result(analysis["stylistic_features"])


def build_faiss_index_interactive() -> None:
    """Build the FAISS index interactively."""
    from fake_news_detector.rag.vector_store import get_vector_store
    print_header("Build FAISS Index")
    limit = prompt_int("Số lượng mẫu train để index", 1000)
    documents = build_documents_from_dataset(limit)
    print(f"Sẽ index {len(documents)} documents vào {settings.faiss_index_path}")

    if not prompt_yes_no("Tiếp tục build index?", default=True):
        print("Đã hủy build index.")
        return

    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    vector_store.save(settings.faiss_index_path)
    print("Build index hoàn tất.")


def query_faiss_index_interactive() -> None:
    """Query the FAISS vector store."""
    from fake_news_detector.rag.vector_store import get_vector_store
    print_header("Query FAISS Index")
    query = prompt("Nhập query")
    k = prompt_int("Số kết quả", 5)

    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)
    if not results:
        print("Không có kết quả. Hãy build index trước.")
        return

    print_header("FAISS Results")
    for item in results:
        print(f"Rank   : {item.get('rank', '?')}")
        print(f"Score  : {item.get('score', 0.0):.3f}")
        print(f"Label  : {item.get('label', '')}")
        print(f"Source : {item.get('source', '')}")
        if item.get("claim"):
            print(f"Claim  : {truncate(item.get('claim', ''), 180)}")
        print(f"Content: {truncate(item.get('content', ''), 320)}")
        print()


def run_agent_pipeline_for_text(text: str) -> None:
    """Run each agent step by step for a text input."""
    print_header("Claim Extractor")
    claims = run_claim_extractor_agent_sync(text)
    print_json({"claims": claims})
    if not claims:
        return

    chosen_index = prompt_int("Chọn claim để tiếp tục", 1) - 1
    if chosen_index < 0 or chosen_index >= len(claims):
        print("Claim index không hợp lệ.")
        return

    claim = claims[chosen_index]
    top_k = prompt_int("Số evidence passages", 5)

    print_header("Evidence Retrieval")
    evidence = run_evidence_retrieval_agent_sync(claim, top_k=top_k)
    print_json(evidence)

    print_header("Verifier")
    verdict = run_verifier_agent_sync(claim, evidence)
    print_json(verdict)

    print_header("Explainer")
    report = run_explainer_agent_sync(claim, verdict, evidence)
    print_json(report)


def test_individual_agents() -> None:
    """Interactive submenu for agent testing."""
    print_header("Agent Testing")
    text = get_article_input()
    if not text:
        print("Không có nội dung để phân tích.")
        return
    run_agent_pipeline_for_text(text)


def run_golden_sample_checks() -> None:
    """Run a small set of golden sample checks."""
    print_header("Golden Sample Checks")
    skip_evidence = prompt_yes_no("Bỏ qua evidence để chạy nhanh?", default=False)
    top_k_evidence = prompt_int("Top-k evidence", 5)

    passed = 0
    for sample in GOLDEN_SAMPLES:
        print(f"\nRunning {sample['id']}...")
        try:
            result = run_trust_only(
                sample["text"],
                skip_evidence=skip_evidence,
                top_k_evidence=top_k_evidence,
            )
            actual = "uncertain"
            if result.results:
                actual = result.results[0].get("verdict", "uncertain")
            expected = sample["expected_verdict"]
            ok = actual == expected
            if ok:
                passed += 1
            print(
                f"Expected: {expected} | Actual: {actual} | "
                f"Status: {'PASS' if ok else 'FAIL'}"
            )
        except Exception as exc:
            print(f"Lỗi khi chạy sample {sample['id']}: {exc}")

    print(f"\nKết quả: {passed}/{len(GOLDEN_SAMPLES)} mẫu khớp expected verdict.")


def full_analysis_interactive() -> None:
    """Run the full analysis flow interactively."""
    article = get_article_input()
    if not article:
        print("Không có nội dung để phân tích.")
        return

    skip_evidence = prompt_yes_no("Bỏ qua evidence retrieval?", default=False)
    top_k_evidence = prompt_int("Top-k evidence", 5)
    analysis = run_full_analysis(article, skip_evidence=skip_evidence, top_k_evidence=top_k_evidence)

    if analysis["summarized"]:
        print("Nội dung dài đã được rút gọn trước khi phân tích.")

    print_baseline_result(analysis["baseline"])
    print_trust_result(analysis["trust"])
    print_stylistic_result(analysis["stylistic_features"])

    if prompt_yes_no("In raw JSON?", default=False):
        print_json(analysis)


def baseline_only_interactive() -> None:
    """Run the baseline-only flow interactively."""
    article = get_article_input()
    if not article:
        print("Không có nội dung để phân tích.")
        return
    print_baseline_result(run_baseline_only(article))


def trust_only_interactive() -> None:
    """Run the TRUST-only flow interactively."""
    article = get_article_input()
    if not article:
        print("Không có nội dung để phân tích.")
        return
    skip_evidence = prompt_yes_no("Bỏ qua evidence retrieval?", default=False)
    top_k_evidence = prompt_int("Top-k evidence", 5)
    result = run_trust_only(article, skip_evidence=skip_evidence, top_k_evidence=top_k_evidence)
    print_trust_result(
        {
            "claims": result.claims,
            "results": result.results,
            "summary": result.summary,
        }
    )


def stylistic_only_interactive() -> None:
    """Run the stylistic analysis interactively."""
    article = get_article_input()
    if not article:
        print("Không có nội dung để phân tích.")
        return
    print_stylistic_result(extract_stylistic_features(article))


def show_menu() -> None:
    """Show the main menu."""
    print(
        "\nChọn chức năng:\n"
        "1. Full analysis\n"
        "2. Baseline only\n"
        "3. TRUST pipeline only\n"
        "4. Stylistic features\n"
        "5. Browse ViFactCheck dataset\n"
        "6. Build FAISS index\n"
        "7. Query FAISS index\n"
        "8. Test individual agents\n"
        "9. Run golden sample checks\n"
        "0. Thoát"
    )


def interactive_loop() -> None:
    """Run the main interactive loop."""
    while True:
        show_menu()
        choice = prompt("Nhập lựa chọn", "0")

        if choice == "1":
            full_analysis_interactive()
        elif choice == "2":
            baseline_only_interactive()
        elif choice == "3":
            trust_only_interactive()
        elif choice == "4":
            stylistic_only_interactive()
        elif choice == "5":
            browse_dataset_samples()
        elif choice == "6":
            build_faiss_index_interactive()
        elif choice == "7":
            query_faiss_index_interactive()
        elif choice == "8":
            test_individual_agents()
        elif choice == "9":
            run_golden_sample_checks()
        elif choice == "0":
            print("Tạm biệt.")
            break
        else:
            print("Lựa chọn không hợp lệ.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Interactive fake news project runner")
    parser.add_argument("--text", help="Run full analysis directly for a text input")
    parser.add_argument("--skip-evidence", action="store_true", help="Skip evidence retrieval in TRUST")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k evidence for TRUST retrieval")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--print-json", action="store_true", help="Print raw JSON for direct analysis")
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    configure_logging(args.verbose)
    startup_checks()

    if args.text:
        analysis = run_full_analysis(
            args.text,
            skip_evidence=args.skip_evidence,
            top_k_evidence=args.top_k,
        )
        print_baseline_result(analysis["baseline"])
        print_trust_result(analysis["trust"])
        print_stylistic_result(analysis["stylistic_features"])
        if args.print_json:
            print_json(analysis)
        return

    interactive_loop()


if __name__ == "__main__":
    main()
