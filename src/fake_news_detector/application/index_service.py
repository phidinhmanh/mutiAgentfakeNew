"""Compatibility wrapper for trust_agents.application.index_service."""

from trust_agents.application.index_service import (  # noqa: F403
    build_vector_index,
    load_sample_claim_and_evidence,
)

__all__ = [
    "load_sample_claim_and_evidence",
    "build_vector_index",
]
