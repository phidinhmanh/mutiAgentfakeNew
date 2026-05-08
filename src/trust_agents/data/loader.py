"""Data loader for Vietnamese Fact-Check dataset."""

import logging
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "tranthaihoa/vifactcheck"


def load_vifactcheck(split: str = "train") -> Any:
    """Load ViFactCheck dataset from HuggingFace.

    Args:
        split: Dataset split to load (train, validation, test)

    Returns:
        HuggingFace Dataset object
    """
    logger.info(f"Loading ViFactCheck dataset: {split}")
    dataset = load_dataset(DATASET_NAME, split=split)
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def get_sample(index: int = 0, split: str = "train") -> dict[str, Any]:
    """Get a single sample from the dataset.

    Args:
        index: Sample index
        split: Dataset split

    Returns:
        Dictionary with claim, evidence, and label
    """
    dataset = load_vifactcheck(split)
    sample = dataset[index]

    return {
        "claim": sample.get("claim", ""),
        "evidence": sample.get("evidence", ""),
        "label": sample.get("label", ""),
        "claim_date": sample.get("claim_date", ""),
        "source": sample.get("source", ""),
    }


def format_sample(sample: dict[str, Any]) -> str:
    """Format a sample for display.

    Args:
        sample: Sample dictionary

    Returns:
        Formatted string
    """
    return f"""Claim: {sample["claim"]}
Evidence: {sample["evidence"]}
Label: {sample["label"]}
Source: {sample.get("source", "N/A")}
Date: {sample.get("claim_date", "N/A")}
"""
