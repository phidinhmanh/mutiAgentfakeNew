"""Benchmark comparison: Baseline vs Single-Agent vs Multi-Agent (Fake) vs TRUST.

TRUST Orchestrator benchmark runs the real 4-agent pipeline:
1. Claim Extractor -> 2. Evidence Retriever -> 3. Verifier -> 4. Explainer

Note: LLM-based approaches may timeout in some environments. This benchmark
runs what it can and reports representative comparison results.
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# Set Groq API key for TRUST orchestrator
os.environ["LLM_PROVIDER"] = "nvidia"
os.environ["NVIDIA_API_KEY"] = "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"
os.environ["NVIDIA_MODEL"] = "openai/gpt-oss-120b"

from trust_agents.config import LLMConfig, LLMProvider, get_llm_config, set_llm_config
from trust_agents.llm.factory import create_chat_model
from trust_agents.orchestrator import TRUSTOrchestrator

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional at import time for unit tests
    load_dataset = None  # type: ignore[assignment]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _require_load_dataset() -> Any:
    if load_dataset is None:
        raise ImportError("datasets dependency is unavailable in this environment")
    return load_dataset


def normalize_sample_label(raw_label: Any) -> str | None:
    """Normalize ViFactCheck labels for benchmark comparisons."""
    if isinstance(raw_label, int):
        return "FAKE" if raw_label == 1 else "REAL"
    if isinstance(raw_label, str):
        label = raw_label.strip().upper()
        if label == "TRUE":
            return "REAL"
        if label == "FALSE":
            return "FAKE"
        if label in {"REAL", "FAKE"}:
            return label
    return None


def extract_vifactcheck_sample(
    item: dict[str, Any],
    sample_id: int,
) -> dict[str, Any] | None:
    """Extract a normalized benchmark sample from either dataset schema."""
    label = normalize_sample_label(item.get("labels", item.get("label")))
    claim_text = (item.get("Statement") or item.get("claim") or "").strip()
    evidence_text = (item.get("Evidence") or item.get("evidence") or "").strip()

    if not label or not claim_text:
        return None

    return {
        "claim": claim_text,
        "evidence": evidence_text,
        "label": label,
        "id": sample_id,
    }


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    approach: str
    samples: int = 0
    total_time: float = 0.0
    avg_time_per_sample: float = 0.0
    correct: int = 0
    accuracy: float = 0.0
    undecided: int = 0
    undecided_rate: float = 0.0
    predictions: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"
    error_message: str = ""


def load_test_data(num_samples: int = 20) -> list[dict[str, Any]]:
    """Load balanced test samples from ViFactCheck."""
    logger.info("Loading %s test samples...", num_samples)
    dataset = _require_load_dataset()("tranthaihoa/vifactcheck", split="test")

    real_samples: list[dict[str, Any]] = []
    fake_samples: list[dict[str, Any]] = []

    for index, item in enumerate(dataset):
        sample = extract_vifactcheck_sample(item, sample_id=index)
        if sample is None:
            continue

        if sample["label"] == "REAL":
            real_samples.append(sample)
        else:
            fake_samples.append(sample)

    half = num_samples // 2
    selected = [*real_samples[:half], *fake_samples[:half]]

    if len(selected) < num_samples:
        for sample in [*real_samples[half:], *fake_samples[half:]]:
            if len(selected) >= num_samples:
                break
            selected.append(sample)

    real_count = sum(1 for sample in selected if sample["label"] == "REAL")
    fake_count = sum(1 for sample in selected if sample["label"] == "FAKE")
    logger.info(
        "Loaded %s: %s REAL, %s FAKE",
        len(selected),
        real_count,
        fake_count,
    )
    return selected


def load_train_baseline_data(num_samples: int = 500) -> list[dict[str, Any]]:
    """Load training data for the baseline keyword model."""
    logger.info("Loading %s training samples for baseline...", num_samples)
    dataset = _require_load_dataset()("tranthaihoa/vifactcheck", split="train")

    samples = []
    for index, item in enumerate(dataset):
        if index >= num_samples:
            break

        sample = extract_vifactcheck_sample(item, sample_id=index)
        if sample is None:
            continue

        samples.append(
            {
                "claim": sample["claim"],
                "evidence": sample["evidence"],
                "label": 1 if sample["label"] == "FAKE" else 0,
            }
        )

    logger.info("Loaded %s training samples", len(samples))
    return samples


def run_baseline_benchmark(
    test_data: list[dict[str, Any]],
    train_data: list[dict[str, Any]],
) -> BenchmarkResult:
    """Run baseline keyword-overlap benchmark."""
    logger.info("Running Baseline benchmark...")
    start_time = time.time()

    fake_keywords: set[str] = set()
    real_keywords: set[str] = set()

    for sample in train_data:
        text = sample["claim"].lower()
        if sample["label"] == 1 or sample["label"] == "FAKE":
            fake_keywords.update(text.split())
        else:
            real_keywords.update(text.split())

    fake_count = sum(
        1 for sample in train_data if sample["label"] == 1 or sample["label"] == "FAKE"
    )
    real_count = len(train_data) - fake_count
    prior_fake = fake_count / len(train_data)
    prior_real = real_count / len(train_data)

    predictions = []
    correct = 0

    for sample in test_data:
        words = set(sample["claim"].lower().split())
        fake_overlap = len(words & fake_keywords)
        real_overlap = len(words & real_keywords)

        fake_score = fake_overlap / max(len(words), 1) + prior_fake * 0.3
        real_score = real_overlap / max(len(words), 1) + prior_real * 0.3
        pred_label = "FAKE" if fake_score > real_score else "REAL"

        is_correct = pred_label == sample["label"]
        if is_correct:
            correct += 1

        predictions.append(_mark_prediction(sample, pred_label, is_correct))

    total_time = time.time() - start_time
    return BenchmarkResult(
        approach="Baseline (Keyword-based)",
        samples=len(test_data),
        total_time=total_time,
        avg_time_per_sample=total_time / len(test_data),
        correct=correct,
        accuracy=correct / len(test_data),
        predictions=predictions,
        status="completed",
    )


def parse_verdict_label(result_text: str) -> str:
    """Extract benchmark verdict labels from free-form model text."""
    upper_text = result_text.upper()
    if "UNVERIFIABLE" in upper_text or "UNCERTAIN" in upper_text:
        return "UNCERTAIN"
    if "FAKE" in upper_text:
        return "FAKE"
    if "REAL" in upper_text:
        return "REAL"
    return "UNKNOWN"


def build_multi_agent_prompt(sample: dict[str, Any]) -> str:
    """Build the benchmark prompt for the evidence-aware path."""
    evidence_text = sample.get("evidence", "")
    evidence_formatted = (
        f"[0] {evidence_text}"
        if evidence_text
        else "[0] Khong co bang chung."
    )

    return f"""Claim: {sample['claim']}

Evidence:
{evidence_formatted}

Tra ve JSON voi dinh dang:
{{
  "verdict": "REAL|FAKE|UNVERIFIABLE",
  "confidence": 0.0-1.0,
  "reasoning": "Giai thich dua tren evidence"
}}

Chi su dung thong tin trong Evidence de giai thich."""


def extract_multi_agent_verdict(result_text: str) -> str:
    """Extract verdict label from the evidence-aware model response."""
    import re

    json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError:
            return parse_verdict_label(result_text)
        verdict = str(result.get("verdict", "UNKNOWN")).upper()
        return parse_verdict_label(verdict)

    return parse_verdict_label(result_text)


def get_benchmark_model_name() -> str:
    """Expose the configured model name for logs and tests."""
    return get_llm_config().model


def get_benchmark_provider_name() -> str:
    """Expose the configured provider name for logs and tests."""
    return get_llm_config().provider.value


def _normalize_benchmark_target_label(label: str) -> str:
    """Keep benchmark labels literal so undecided outputs remain visible."""
    return label


def _select_best_trust_result(
    results: list[dict[str, Any]],
    source_claim: str,
) -> dict[str, Any]:
    """Pick the claim result most lexically similar to the source benchmark claim.

    Falls back to index 0 if no results are available.
    """
    if not results:
        return {}

    source_words = set(source_claim.lower().split())
    best_idx = 0
    best_overlap = -1

    for idx, result in enumerate(results):
        claim_text = result.get("claim", "")
        claim_words = set(claim_text.lower().split())
        overlap = len(source_words & claim_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    return results[best_idx]


def _score_prediction(predicted_label: str, expected_label: str) -> tuple[bool, bool]:
    """Score benchmark predictions against binary ViFactCheck labels.

    Returns:
        (is_correct, is_undecided). is_undecided is True when the prediction
        was UNCERTAIN so it can be tracked separately from wrong predictions.
    """
    pred = _normalize_benchmark_target_label(predicted_label)
    exp = _normalize_benchmark_target_label(expected_label)
    if pred == "UNCERTAIN" or exp == "UNCERTAIN":
        return (False, pred == "UNCERTAIN")
    return (pred == exp, False)


def _build_single_agent_approach_name() -> str:
    provider = get_benchmark_provider_name()
    model = get_benchmark_model_name()
    return f"Single-Agent ({provider}:{model})"


def _build_multi_agent_approach_name() -> str:
    provider = get_benchmark_provider_name()
    model = get_benchmark_model_name()
    return f"Multi-Agent ({provider}:{model})"


def _invoke_multi_agent_model(llm: Any, sample: dict[str, Any]) -> str:
    return invoke_text_model(
        llm,
        system_prompt="Ban la chuyen gia xac thuc tin tuc Viet Nam.",
        user_prompt=build_multi_agent_prompt(sample),
    )


def _invoke_single_agent_model(
    llm: Any,
    sample: dict[str, Any],
    system_prompt: str,
) -> str:
    return invoke_text_model(
        llm,
        system_prompt=system_prompt,
        user_prompt=f"Tin sau la that hay gia: {sample['claim']}",
    )


def _compute_accuracy(correct: int, samples_processed: int) -> float:
    return correct / samples_processed if samples_processed > 0 else 0


def _compute_avg_time(total_time: float, samples_processed: int) -> float:
    return total_time / samples_processed if samples_processed > 0 else 0


def _mark_prediction(
    sample: dict[str, Any],
    pred_label: str,
    is_correct: bool,
) -> dict[str, Any]:
    return {
        "id": sample["id"],
        "claim": sample["claim"][:80],
        "true_label": sample["label"],
        "predicted_label": pred_label,
        "correct": is_correct,
    }


def _update_error(
    current_status: str,
    current_error: str,
    exc: Exception,
) -> tuple[str, str]:
    del current_status
    logger.warning("Benchmark model error: %s", exc)
    return "error", current_error or str(exc)[:100]


def _prepare_predictions_result(
    approach: str,
    start_time: float,
    predictions: list[dict[str, Any]],
    correct: int,
    undecided: int,
    status: str,
    error_msg: str,
) -> BenchmarkResult:
    total_time = time.time() - start_time
    samples_processed = len(predictions)
    return BenchmarkResult(
        approach=approach,
        samples=samples_processed,
        total_time=total_time,
        avg_time_per_sample=_compute_avg_time(total_time, samples_processed),
        correct=correct,
        accuracy=_compute_accuracy(correct, samples_processed),
        undecided=undecided,
        undecided_rate=_compute_accuracy(undecided, samples_processed),
        predictions=predictions,
        status=status,
        error_message=error_msg,
    )


def _predict_single_sample(llm: Any, sample: dict[str, Any], system_prompt: str) -> str:
    result_text = _invoke_single_agent_model(llm, sample, system_prompt).strip().upper()
    return parse_verdict_label(result_text)


def _predict_multi_sample(llm: Any, sample: dict[str, Any]) -> str:
    result_text = _invoke_multi_agent_model(llm, sample).strip()
    return extract_multi_agent_verdict(result_text)


def _score_and_store_prediction(
    predictions: list[dict[str, Any]],
    sample: dict[str, Any],
    pred_label: str,
) -> tuple[int, int]:
    is_correct, is_undecided = _score_prediction(pred_label, sample["label"])
    predictions.append(_mark_prediction(sample, pred_label, is_correct))
    return (1 if is_correct else 0, 1 if is_undecided else 0)


def _openai_like_model(llm: Any) -> bool:
    chat = getattr(llm, "chat", None)
    completions = getattr(chat, "completions", None)
    create = getattr(completions, "create", None)
    return callable(create)


def _provider_supports_reasoning_content() -> bool:
    return get_llm_config().provider == LLMProvider.GEMINI_NVIDIA


def _extract_openai_message_text(message: Any) -> str:
    if _provider_supports_reasoning_content():
        return message.content or getattr(message, "reasoning_content", "") or ""
    return message.content or ""


def invoke_text_model(llm: Any, system_prompt: str, user_prompt: str) -> str:
    """Invoke either OpenAI-compatible or LangChain-like chat models."""
    if _openai_like_model(llm):
        config = get_llm_config()
        response = llm.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
            max_tokens=min(config.max_tokens, 1000),
        )
        return _extract_openai_message_text(response.choices[0].message)

    invoke = getattr(llm, "invoke", None)
    if callable(invoke):
        response = invoke([
            ("system", system_prompt),
            ("human", user_prompt),
        ])
        return getattr(response, "content", str(response))

    raise TypeError(f"Unsupported benchmark model type: {type(llm).__name__}")

def create_benchmark_model(timeout_seconds: int) -> Any:
    """Create the LLM used by benchmark flows from shared TRUST config."""
    config = get_llm_config().model_copy()

    if config.provider == LLMProvider.OPENAI:
        import os

        from openai import OpenAI

        api_key = config.get_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        base_url = os.getenv("OPENAI_BASE_URL") or None
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout_seconds),
        )

    if config.provider == LLMProvider.GEMINI_NVIDIA:
        import os

        from openai import OpenAI

        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable required")
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=float(timeout_seconds),
        )

    return create_chat_model(config)


def run_multi_agent_benchmark(
    test_data: list[dict[str, Any]],
    timeout_seconds: int = 60,
) -> BenchmarkResult:
    """Run multi-agent benchmark with the configured provider."""
    logger.info(
        "Running Multi-Agent benchmark with %ss per-request timeout...",
        timeout_seconds,
    )
    start_time = time.time()
    predictions = []
    correct = 0
    undecided = 0
    status = "completed"
    error_msg = ""
    llm = create_benchmark_model(timeout_seconds)

    for sample in test_data:
        try:
            pred_label = _predict_multi_sample(llm, sample)
        except Exception as exc:
            status, error_msg = _update_error(status, error_msg, exc)
            pred_label = "ERROR"

        c, u = _score_and_store_prediction(predictions, sample, pred_label)
        correct += c
        undecided += u
        # Rate limit protection for Groq (8,000 TPM is tight)
        if get_benchmark_provider_name() == "groq":
            time.sleep(5)

    return _prepare_predictions_result(
        approach=_build_multi_agent_approach_name(),
        start_time=start_time,
        predictions=predictions,
        correct=correct,
        undecided=undecided,
        status=status,
        error_msg=error_msg,
    )


def run_single_agent_benchmark(
    test_data: list[dict[str, Any]],
    timeout_seconds: int = 60,
) -> BenchmarkResult:
    """Run single-agent benchmark with the configured provider."""
    logger.info(
        "Running Single-Agent benchmark with %ss per-request timeout...",
        timeout_seconds,
    )
    start_time = time.time()
    predictions = []
    correct = 0
    undecided = 0
    status = "completed"
    error_msg = ""

    system_prompt = """Ban la chuyen gia xac thuc tin tuc Viet Nam.
Tra ve CHI MOT TU: REAL neu tin that, FAKE neu tin gia.
Khong giai thich, chi tra ve REAL hoac FAKE."""
    llm = create_benchmark_model(timeout_seconds)

    for sample in test_data:
        try:
            pred_label = _predict_single_sample(llm, sample, system_prompt)
        except Exception as exc:
            status, error_msg = _update_error(status, error_msg, exc)
            pred_label = "ERROR"

        c, u = _score_and_store_prediction(predictions, sample, pred_label)
        correct += c
        undecided += u
        if get_benchmark_provider_name() == "groq":
            time.sleep(5)

    return _prepare_predictions_result(
        approach=_build_single_agent_approach_name(),
        start_time=start_time,
        predictions=predictions,
        correct=correct,
        undecided=undecided,
        status=status,
        error_msg=error_msg,
    )


def _normalize_trust_label(trust_label: str) -> str:
    """Normalize TRUST verdict label to benchmark format (REAL/FAKE).

    Only decisive verdicts are mapped. Uncertain/unverifiable pass through as
    "UNCERTAIN" so benchmark reporting can track abstention separately.
    """
    label_upper = trust_label.upper()
    if label_upper == "TRUE":
        return "REAL"
    elif label_upper == "FALSE":
        return "FAKE"
    return "UNCERTAIN"


def run_trust_orchestrator_benchmark(
    test_data: list[dict[str, Any]],
    timeout_seconds: int = 60,
) -> BenchmarkResult:
    """Run TRUST Orchestrator benchmark with real multi-agent pipeline."""
    logger.info(
        "Running TRUST Orchestrator benchmark with %ss per-request timeout...",
        timeout_seconds,
    )

    nvidia_config = LLMConfig(
        provider=LLMProvider.GEMINI_NVIDIA,
        model=os.getenv("NVIDIA_MODEL", "openai/gpt-oss-120b"),
        temperature=0.1,
        max_tokens=2048,
    )
    set_llm_config(nvidia_config)

    start_time = time.time()
    predictions = []
    correct = 0
    undecided = 0
    status = "completed"
    error_msg = ""

    orchestrator = TRUSTOrchestrator(top_k_evidence=5)

    for sample in test_data:
        try:
            # Benchmark fairness: keep input as the source claim only.
            text = sample["claim"]

            result = orchestrator.process_text(text, skip_evidence=False)

            # Extract verdict from best-matching claim, not blindly index 0.
            pred_label = "UNKNOWN"
            if result.results:
                best_result = _select_best_trust_result(result.results, sample["claim"])
                trust_verdict = best_result.get("verdict", "uncertain")
                pred_label = _normalize_trust_label(trust_verdict)

        except Exception as exc:
            status, error_msg = _update_error(status, error_msg, exc)
            pred_label = "ERROR"

        c, u = _score_and_store_prediction(predictions, sample, pred_label)
        correct += c
        undecided += u
        # Rate limit protection for Groq (8,000 TPM is tight)
        if get_benchmark_provider_name() == "groq":
            time.sleep(5)

    return _prepare_predictions_result(
        approach="TRUST Orchestrator (Real Multi-Agent)",
        start_time=start_time,
        predictions=predictions,
        correct=correct,
        undecided=undecided,
        status=status,
        error_msg=error_msg,
    )


def get_nvidia_client() -> Any:
    """Backward-compatible helper for NVIDIA benchmark callers."""
    import os

    from openai import OpenAI

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable required")

    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results."""
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: Baseline vs Single-Agent vs Multi-Agent (Fake) vs TRUST")
    print("=" * 80)

    test_samples = results[0].samples if results else 0
    print(f"\nTest samples: {test_samples}")
    print("Training samples for baseline: 500")
    print("\n" + "-" * 80)

    for result in results:
        status_suffix = (
            f" [{result.status.upper()}]" if result.status != "completed" else ""
        )
        print(f"\n{result.approach}{status_suffix}")
        print(
            f"  Accuracy:     {result.accuracy:.1%} "
            f"({result.correct}/{result.samples})"
        )
        print(f"  Undecided:    {result.undecided_rate:.1%} "
              f"({result.undecided}/{result.samples})")
        print(f"  Total time:   {result.total_time:.2f}s")
        print(f"  Avg/sample:   {result.avg_time_per_sample:.2f}s")
        if result.error_message:
            print(f"  Error:        {result.error_message}")

    print("\n" + "-" * 80)
    print("\nDetailed Results:")
    print("-" * 80)

    for result in results:
        print(f"\n=== {result.approach} ===")
        for prediction in result.predictions[:5]:
            status = "OK" if prediction["correct"] else "FAIL"
            claim_preview = prediction["claim"][:50] if prediction["claim"] else "empty"
            print(
                f"  [{status}] ID={prediction['id']:2d} "
                f"| True={prediction['true_label']:4s} "
                f"| Pred={prediction['predicted_label']:4s} "
                f"| {claim_preview}..."
            )
        if len(result.predictions) > 5:
            print(f"  ... and {len(result.predictions) - 5} more")


def save_results(results: list[BenchmarkResult], output_path: str) -> None:
    """Save results to JSON file."""
    output = {
        "benchmark": (
            "Baseline vs Single-Agent vs Multi-Agent (Fake) vs TRUST Orchestrator"
        ),
        "test_samples": results[0].samples if results else 0,
        "train_samples_baseline": 500,
        "results": [asdict(result) for result in results],
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", output_path)


def main() -> None:
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark fake news detection")
    parser.add_argument(
        "--test-samples",
        type=int,
        default=20,
        help="Number of test samples",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=500,
        help="Number of training samples for baseline",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=30,
        help="Per-request timeout in seconds for LLM calls",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file",
    )
    parser.add_argument(
        "--skip-single",
        action="store_true",
        help="Skip single-agent benchmark",
    )
    parser.add_argument(
        "--skip-multi",
        action="store_true",
        help="Skip multi-agent (fake) benchmark",
    )
    parser.add_argument(
        "--skip-trust",
        action="store_true",
        help="Skip real TRUST orchestrator benchmark",
    )
    args = parser.parse_args()

    test_data = load_test_data(args.test_samples)
    train_data = load_train_baseline_data(args.train_samples)

    initial_config = LLMConfig(
        provider=LLMProvider.GEMINI_NVIDIA,
        model=os.getenv("NVIDIA_MODEL", "openai/gpt-oss-120b"),
        temperature=0.1,
        max_tokens=2048,
    )
    set_llm_config(initial_config)

    results = [run_baseline_benchmark(test_data, train_data)]

    if not args.skip_single:
        try:
            single_result = run_single_agent_benchmark(
                test_data,
                timeout_seconds=args.llm_timeout,
            )
            results.append(single_result)
        except Exception as exc:
            logger.error("Single-agent benchmark failed: %s", exc)

    if not args.skip_multi:
        try:
            multi_result = run_multi_agent_benchmark(
                test_data,
                timeout_seconds=args.llm_timeout,
            )
            results.append(multi_result)
        except Exception as exc:
            logger.error("Multi-agent (fake) benchmark failed: %s", exc)

    if not args.skip_trust:
        try:
            trust_result = run_trust_orchestrator_benchmark(
                test_data,
                timeout_seconds=args.llm_timeout,
            )
            results.append(trust_result)
        except Exception as exc:
            logger.error("TRUST orchestrator benchmark failed: %s", exc)

    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
