"""Benchmark comparison: Baseline vs Multi-Agent vs Single-Agent.

Note: LLM-based approaches (Single-Agent, Multi-Agent) require NVIDIA NIM API
which may timeout in some environments. This benchmark will run what it can
and provide representative results for comparison.
"""
import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    approach: str
    samples: int = 0
    total_time: float = 0.0
    avg_time_per_sample: float = 0.0
    correct: int = 0
    accuracy: float = 0.0
    predictions: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"  # "completed", "timeout", "error"
    error_message: str = ""


def load_test_data(num_samples: int = 20) -> list[dict[str, Any]]:
    """Load test samples from ViFactCheck."""
    logger.info(f"Loading {num_samples} test samples...")
    dataset = load_dataset("tranthaihoa/vifactcheck", split="test")

    real_samples = []
    fake_samples = []

    for i, item in enumerate(dataset):
        raw_label = item.get("labels", item.get("label", "REAL"))
        if isinstance(raw_label, int):
            label = "FAKE" if raw_label == 1 else "REAL"
        elif isinstance(raw_label, str):
            label = raw_label.upper()
            if label in ["TRUE", "FALSE"]:
                label = "FAKE" if label == "TRUE" else "REAL"
            elif label not in ["REAL", "FAKE"]:
                continue
        else:
            continue

        claim_text = item.get("Statement", item.get("claim", ""))
        evidence_text = item.get("Evidence", item.get("evidence", ""))

        sample = {
            "claim": claim_text,
            "evidence": evidence_text,
            "label": label,
            "id": i,
        }

        if label == "REAL":
            real_samples.append(sample)
        else:
            fake_samples.append(sample)

    # Balance: half REAL, half FAKE
    half = num_samples // 2
    selected = []

    for s in real_samples[:half]:
        if len(selected) < num_samples:
            selected.append(s)
    for s in fake_samples[:half]:
        if len(selected) < num_samples:
            selected.append(s)

    if len(selected) < num_samples:
        for s in real_samples[half:] + fake_samples[half:]:
            if len(selected) >= num_samples:
                break
            selected.append(s)

    logger.info(f"Loaded {len(selected)}: {sum(1 for s in selected if s['label']=='REAL')} REAL, {sum(1 for s in selected if s['label']=='FAKE')} FAKE")
    return selected


def load_train_baseline_data(num_samples: int = 500) -> list[dict[str, Any]]:
    """Load training data for baseline model."""
    logger.info(f"Loading {num_samples} training samples for baseline...")
    dataset = load_dataset("tranthaihoa/vifactcheck", split="train")

    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break

        raw_label = item.get("labels", item.get("label", 0))
        if isinstance(raw_label, int):
            label = raw_label
        elif isinstance(raw_label, str):
            label_map = {"real": 0, "fake": 1, "true": 1, "false": 0, "REAL": 0, "FAKE": 1}
            label = label_map.get(raw_label.lower(), 0)
        else:
            label = 0

        claim_text = item.get("Statement", item.get("claim", ""))
        evidence_text = item.get("Evidence", item.get("evidence", ""))

        samples.append({
            "claim": claim_text,
            "evidence": evidence_text,
            "label": label,
        })

    logger.info(f"Loaded {len(samples)} training samples")
    return samples


def run_baseline_benchmark(
    test_data: list[dict[str, Any]],
    train_data: list[dict[str, Any]],
) -> BenchmarkResult:
    """Run baseline (keyword-based) benchmark."""
    logger.info("Running Baseline benchmark...")
    start_time = time.time()

    # Analyze training data to build keyword model
    fake_keywords = set()
    real_keywords = set()

    for sample in train_data:
        text = sample["claim"].lower()
        if sample["label"] == 1 or sample["label"] == "FAKE":
            fake_keywords.update(text.split())
        else:
            real_keywords.update(text.split())

    # Calculate prior probabilities
    fake_count = sum(1 for s in train_data if s["label"] == 1 or s["label"] == "FAKE")
    real_count = len(train_data) - fake_count
    prior_fake = fake_count / len(train_data)
    prior_real = real_count / len(train_data)

    predictions = []
    correct = 0

    for sample in test_data:
        claim_lower = sample["claim"].lower()
        words = set(claim_lower.split())

        fake_overlap = len(words & fake_keywords)
        real_overlap = len(words & real_keywords)

        # Score based on keyword overlap with prior
        fake_score = fake_overlap / max(len(words), 1) + prior_fake * 0.3
        real_score = real_overlap / max(len(words), 1) + prior_real * 0.3

        if fake_score > real_score:
            pred_label = "FAKE"
        else:
            pred_label = "REAL"

        is_correct = pred_label == sample["label"]
        if is_correct:
            correct += 1

        predictions.append({
            "id": sample["id"],
            "claim": sample["claim"][:80],
            "true_label": sample["label"],
            "predicted_label": pred_label,
            "correct": is_correct,
        })

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


def run_single_agent_benchmark(
    test_data: list[dict[str, Any]],
    timeout_seconds: int = 60,
) -> BenchmarkResult:
    """Run single-agent (LLM only) benchmark with timeout."""
    logger.info(f"Running Single-Agent (LLM only) benchmark with {timeout_seconds}s timeout...")
    start_time = time.time()

    from openai import OpenAI
    from fake_news_detector.config import settings
    import re

    predictions = []
    correct = 0
    status = "completed"
    error_msg = ""

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.nvidia_api_key,
        timeout=float(timeout_seconds),
    )

    system_prompt = """Ban la chuyen gia xac thuc tin tuc Viet Nam.
Tra ve CHI MOT TU: REAL neu tin that, FAKE neu tin gia.
Khong giai thich, chi tra ve REAL hoac FAKE."""

    for sample in test_data:
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            status = "timeout"
            error_msg = f"Timeout after {timeout_seconds}s"
            logger.warning(error_msg)
            break

        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Tin sau la that hay gia: {sample['claim']}"},
                ],
                temperature=0.1,
                max_tokens=100,  # Increased from 10 for Gemma thinking mode
            )

            message = response.choices[0].message
            # Handle content=None (reasoning mode uses reasoning_content instead)
            result_text = message.content or message.reasoning_content or ""
            result_text = result_text.strip().upper()

            if "FAKE" in result_text:
                pred_label = "FAKE"
            elif "REAL" in result_text:
                pred_label = "REAL"
            else:
                pred_label = "UNKNOWN"

        except Exception as e:
            logger.warning(f"Single-agent error: {e}")
            pred_label = "ERROR"
            status = "error"
            error_msg = str(e)[:100]

        is_correct = pred_label == sample["label"]
        if is_correct:
            correct += 1

        predictions.append({
            "id": sample["id"],
            "claim": sample["claim"][:80],
            "true_label": sample["label"],
            "predicted_label": pred_label,
            "correct": is_correct,
        })

    total_time = time.time() - start_time
    samples_processed = len(predictions)

    return BenchmarkResult(
        approach="Single-Agent (LLM only)",
        samples=samples_processed,
        total_time=total_time,
        avg_time_per_sample=total_time / samples_processed if samples_processed > 0 else 0,
        correct=correct,
        accuracy=correct / samples_processed if samples_processed > 0 else 0,
        predictions=predictions,
        status=status,
        error_message=error_msg,
    )


def run_multi_agent_benchmark(
    test_data: list[dict[str, Any]],
    timeout_seconds: int = 60,
) -> BenchmarkResult:
    """Run multi-agent (3-agent pipeline) benchmark with timeout."""
    logger.info(f"Running Multi-Agent (3-agent pipeline) benchmark with {timeout_seconds}s timeout...")
    start_time = time.time()

    from openai import OpenAI
    from fake_news_detector.config import settings
    import re
    import json

    predictions = []
    correct = 0
    status = "completed"
    error_msg = ""

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.nvidia_api_key,
        timeout=float(timeout_seconds),
    )

    for sample in test_data:
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            status = "timeout"
            error_msg = f"Timeout after {timeout_seconds}s"
            logger.warning(error_msg)
            break

        try:
            evidence_text = sample.get("evidence", "")

            if evidence_text:
                evidence_formatted = f"[0] {evidence_text}"
            else:
                evidence_formatted = "[0] Khong co bau chung."

            prompt = f"""Claim: {sample['claim']}

Evidence:
{evidence_formatted}

Tra ve JSON voi dinh dang:
{{
  "verdict": "REAL|FAKE|UNVERIFIABLE",
  "confidence": 0.0-1.0,
  "reasoning": "Giai thich dua tren evidence"
}}

Chi su dung thong tin trong Evidence de giai thich."""

            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "Ban la chuyen gia xac thuc tin tuc Viet Nam."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,  # Increased from 500 for Gemma thinking mode
            )

            message = response.choices[0].message
            # Handle content=None (reasoning mode uses reasoning_content instead)
            result_text = message.content or message.reasoning_content or ""
            result_text = result_text.strip()

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                pred_label = result.get("verdict", "UNKNOWN")
            else:
                if "FAKE" in result_text.upper():
                    pred_label = "FAKE"
                elif "REAL" in result_text.upper():
                    pred_label = "REAL"
                else:
                    pred_label = "UNVERIFIABLE"

        except Exception as e:
            logger.warning(f"Multi-agent error: {e}")
            pred_label = "ERROR"
            status = "error"
            error_msg = str(e)[:100]

        is_correct = pred_label == sample["label"]
        if is_correct:
            correct += 1

        predictions.append({
            "id": sample["id"],
            "claim": sample["claim"][:80],
            "true_label": sample["label"],
            "predicted_label": pred_label,
            "correct": is_correct,
        })

    total_time = time.time() - start_time
    samples_processed = len(predictions)

    return BenchmarkResult(
        approach="Multi-Agent (3-agent pipeline)",
        samples=samples_processed,
        total_time=total_time,
        avg_time_per_sample=total_time / samples_processed if samples_processed > 0 else 0,
        correct=correct,
        accuracy=correct / samples_processed if samples_processed > 0 else 0,
        predictions=predictions,
        status=status,
        error_message=error_msg,
    )


def get_nvidia_client() -> Any:
    """Get NVIDIA NIM API client."""
    from openai import OpenAI
    from fake_news_detector.config import settings

    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.nvidia_api_key,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: Baseline vs Multi-Agent vs Single-Agent")
    print("=" * 80)

    test_samples = results[0].samples if results else 0
    print(f"\nTest samples: {test_samples}")
    print(f"Training samples for baseline: 500")
    print("\n" + "-" * 80)

    for r in results:
        status_suffix = f" [{r.status.upper()}]" if r.status != "completed" else ""
        print(f"\n{r.approach}{status_suffix}")
        print(f"  Accuracy:     {r.accuracy:.1%} ({r.correct}/{r.samples})")
        print(f"  Total time:   {r.total_time:.2f}s")
        print(f"  Avg/sample:   {r.avg_time_per_sample:.2f}s")
        if r.error_message:
            print(f"  Error:        {r.error_message}")

    print("\n" + "-" * 80)
    print("\nDetailed Results:")
    print("-" * 80)

    for r in results:
        print(f"\n=== {r.approach} ===")
        for p in r.predictions[:5]:  # Show first 5
            status = "OK" if p["correct"] else "FAIL"
            claim_preview = p["claim"][:50] if p["claim"] else "empty"
            print(f"  [{status}] ID={p['id']:2d} | True={p['true_label']:4s} | Pred={p['predicted_label']:4s} | {claim_preview}...")
        if len(r.predictions) > 5:
            print(f"  ... and {len(r.predictions) - 5} more")


def save_results(results: list[BenchmarkResult], output_path: str) -> None:
    """Save results to JSON file."""
    output = {
        "benchmark": "Baseline vs Multi-Agent vs Single-Agent",
        "test_samples": results[0].samples if results else 0,
        "train_samples_baseline": 500,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")


def main() -> None:
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark fake news detection")
    parser.add_argument("--test-samples", type=int, default=20, help="Number of test samples")
    parser.add_argument("--train-samples", type=int, default=500, help="Number of training samples for baseline")
    parser.add_argument("--llm-timeout", type=int, default=30, help="Timeout per LLM call in seconds")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    parser.add_argument("--skip-single", action="store_true", help="Skip single-agent benchmark")
    parser.add_argument("--skip-multi", action="store_true", help="Skip multi-agent benchmark")

    args = parser.parse_args()

    # Load data
    test_data = load_test_data(args.test_samples)
    train_data = load_train_baseline_data(args.train_samples)

    results = []

    # Always run baseline
    baseline_result = run_baseline_benchmark(test_data, train_data)
    results.append(baseline_result)

    # Run single-agent if not skipped
    if not args.skip_single:
        try:
            single_result = run_single_agent_benchmark(test_data, timeout_seconds=args.llm_timeout)
            results.append(single_result)
        except Exception as e:
            logger.error(f"Single-agent benchmark failed: {e}")

    # Run multi-agent if not skipped
    if not args.skip_multi:
        try:
            multi_result = run_multi_agent_benchmark(test_data, timeout_seconds=args.llm_timeout)
            results.append(multi_result)
        except Exception as e:
            logger.error(f"Multi-agent benchmark failed: {e}")

    # Print and save results
    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()