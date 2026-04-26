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


def analyze_with_legacy(
    article: str,
    extract_claims_fn,
    filter_verifiable_claims_fn,
    retrieve_evidence_for_claims_fn,
    enrich_evidence_with_context_fn,
) -> dict[str, Any]:
    """Run baseline, legacy pipeline, and stylistic analysis."""
    processed_article, summarized = _prepare_article(article)
    baseline_model = get_baseline_model()
    baseline_result = baseline_model.predict_with_sliding_window(processed_article)

    claims = extract_claims_fn(processed_article)
    verifiable_claims = filter_verifiable_claims_fn(claims)

    response: dict[str, Any] = {
        "baseline": baseline_result,
        "claims": claims,
        "verifiable_claims": verifiable_claims,
        "stylistic_features": extract_stylistic_features(processed_article),
    }

    if verifiable_claims:
        claims_with_evidence = retrieve_evidence_for_claims_fn(verifiable_claims)
        for claim_with_evidence in claims_with_evidence:
            claim_with_evidence["evidence"] = enrich_evidence_with_context_fn(
                claim_with_evidence.get("evidence", []),
                claim_with_evidence.get("text", ""),
            )

        merged_evidence: list[dict[str, Any]] = []
        for claim_with_evidence in claims_with_evidence:
            merged_evidence.extend(claim_with_evidence.get("evidence", []))

        response["claims_with_evidence"] = claims_with_evidence
        response["evidence"] = merged_evidence[:10]

    if summarized:
        response["summarized"] = True
    return response


def _prepare_article(article: str) -> tuple[str, bool]:
    """Summarize long articles before downstream analysis."""
    if len(article) > 3000:
        return summarize_for_long_text(article, max_chars=3000), True
    return article, False
