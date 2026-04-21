"""Attention visualization for model interpretability."""
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fake_news_detector.data.preprocessing import tokenize_words

logger = logging.getLogger(__name__)


def plot_claim_importance(
    claims: list[dict[str, Any]],
    save_path: str | None = None,
) -> None:
    """Plot importance scores for extracted claims.

    Args:
        claims: List of claims with importance scores
        save_path: Optional path to save the plot
    """
    if not claims:
        logger.warning("No claims to plot")
        return

    claim_texts = [c.get("text", "")[:50] for c in claims]
    fake_probs = [c.get("fake_prob", 0.5) for c in claims]

    fig, ax = plt.subplots(figsize=(10, max(4, len(claims) * 0.5)))
    colors = ["red" if p > 0.5 else "green" for p in fake_probs]

    y_pos = np.arange(len(claim_texts))
    ax.barh(y_pos, fake_probs, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(claim_texts)
    ax.set_xlabel("Fake Probability")
    ax.set_title("Claim Importance Analysis")
    ax.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved attention plot to {save_path}")

    plt.close()


def plot_evidence_scores(
    evidence: list[dict[str, Any]],
    save_path: str | None = None,
) -> None:
    """Plot evidence relevance scores.

    Args:
        evidence: List of evidence items with scores
        save_path: Optional path to save the plot
    """
    if not evidence:
        logger.warning("No evidence to plot")
        return

    sources = [e.get("source", "unknown")[:20] for e in evidence]
    scores = [e.get("score", 0.0) for e in evidence]

    fig, ax = plt.subplots(figsize=(10, max(4, len(evidence) * 0.5)))

    y_pos = np.arange(len(sources))
    ax.barh(y_pos, scores, color="steelblue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sources)
    ax.set_xlabel("Relevance Score")
    ax.set_title("Evidence Relevance Analysis")
    ax.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved evidence plot to {save_path}")

    plt.close()


def plot_confidence_distribution(
    verdicts: list[dict[str, Any]],
    save_path: str | None = None,
) -> None:
    """Plot distribution of confidence scores.

    Args:
        verdicts: List of verdict results
        save_path: Optional path to save the plot
    """
    if not verdicts:
        logger.warning("No verdicts to plot")
        return

    confidences = [v.get("confidence", 0.0) for v in verdicts]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(confidences, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence Distribution")

    verdict_counts = {
        "REAL": sum(1 for v in verdicts if v.get("verdict") == "REAL"),
        "FAKE": sum(1 for v in verdicts if v.get("verdict") == "FAKE"),
        "UNVERIFIABLE": sum(1 for v in verdicts if v.get("verdict") == "UNVERIFIABLE"),
    }

    axes[1].pie(
        verdict_counts.values(),
        labels=verdict_counts.keys(),
        autopct="%1.1f%%",
        colors=["green", "red", "gray"],
    )
    axes[1].set_title("Verdict Distribution")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confidence plot to {save_path}")

    plt.close()


def plot_stylistic_features(
    features: dict[str, Any],
    save_path: str | None = None,
) -> None:
    """Plot stylistic feature analysis.

    Args:
        features: Dictionary of stylistic features
        save_path: Optional path to save the plot
    """
    feature_names = [
        "caps_ratio",
        "emotional_markers",
        "sensational_words",
        "source_mentions",
    ]

    values = [features.get(f, 0) for f in feature_names]

    fig, ax = plt.subplots(figsize=(10, 5))

    x_pos = np.arange(len(feature_names))
    ax.bar(x_pos, values, color="coral", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Count / Ratio")
    ax.set_title("Stylistic Feature Analysis")

    fake_score = features.get("fake_score", 0.5)
    ax.axhline(y=fake_score, color="red", linestyle="--", label=f"Fake Score: {fake_score:.2f}")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved stylistic features plot to {save_path}")

    plt.close()