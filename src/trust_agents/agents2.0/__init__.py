"""
TRUST Agents 2.0 - Research-Grade Components

WARNING: This module is for research and experimental purposes only.
It is NOT recommended for production use as it has not been fully stabilized.

This module contains cutting-edge research components:
- Decomposer Agent (LoCal-inspired)
- Logic Aggregator
- Delphi Jury (Multi-agent system)

Based on 2024-2025 research papers.
"""

from .decomposer_agent import DecomposerAgent, decompose_claim_sync
from .delphi_jury import DelphiJury, verify_with_delphi_jury_sync
from .logic_aggregator import LogicAggregator

__all__ = [
    "DecomposerAgent",
    "decompose_claim_sync",
    "LogicAggregator",
    "DelphiJury",
    "verify_with_delphi_jury_sync",
]
