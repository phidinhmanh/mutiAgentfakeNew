"""Compatibility wrapper for trust_agents.application.analysis_service."""

from trust_agents.application.analysis_service import (  # noqa: F403
    TrustPipelineProtocol,
    _prepare_article,
    analyze_with_trust,
)

__all__ = [
    "TrustPipelineProtocol",
    "_prepare_article",
    "analyze_with_trust",
]
