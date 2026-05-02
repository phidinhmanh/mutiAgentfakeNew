"""Application service for article analysis flows."""

from __future__ import annotations

from typing import Any

from fake_news_detector.data.preprocessing import summarize_for_long_text
from fake_news_detector.models.baseline import get_baseline_model
from fake_news_detector.models.stylistic import extract_stylistic_features


class TrustPipelineProtocol:
    """Minimal protocol-like adapter for TRUST orchestrators."""

    def process_text(self, article: str) -> Any:  # pragma: no cover - duck typed
        raise NotImplementedError


def analyze_with_trust(
    article: str,
    orchestrator: TrustPipelineProtocol,
) -> dict[str, Any]:
    """Run baseline, TRUST pipeline, and stylistic analysis."""
    processed_article, summarized = _prepare_article(article)
    baseline_model = get_baseline_model()
    baseline_result = baseline_model.predict_with_sliding_window(processed_article)

    process_text = getattr(orchestrator, "process_text", None)
    if not callable(process_text):
        raise TypeError("Invalid TRUST orchestrator: missing callable process_text")

    try:
        trust_result = process_text(processed_article)
    except NotImplementedError as exc:
        raise TypeError(
            "Invalid TRUST orchestrator: process_text is not implemented"
        ) from exc

    verdicts = [
        {
            "verdict": result.get("verdict", "uncertain"),
            "confidence": result.get("confidence", 0.3),
            "label": result.get("label", "uncertain"),
            "reasoning": result.get("reasoning", ""),
            "claim": result.get("claim", ""),
        }
        for result in trust_result.results
    ]

    response: dict[str, Any] = {
        "baseline": baseline_result,
        "trust": {
            "claims": trust_result.claims,
            "results": trust_result.results,
            "summary": trust_result.summary,
        },
        "claims": [{"text": claim, "source": "trust"} for claim in trust_result.claims],
        "verdicts": verdicts,
        "stylistic_features": extract_stylistic_features(processed_article),
    }
    if summarized:
        response["summarized"] = True
    return response


    if summarized:
        response["summarized"] = True
    return response


def _prepare_article(article: str) -> tuple[str, bool]:
    """Summarize long articles before downstream analysis."""
    if len(article) > 3000:
        return summarize_for_long_text(article, max_chars=3000), True
    return article, False
