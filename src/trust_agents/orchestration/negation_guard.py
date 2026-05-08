"""Negation detection for fact-checking evidence."""

from shared_fact_checking.constants import NEGATION_KEYWORDS, NEGATION_TRIGGER_MODIFIERS


def negation_scanner(evidence_texts: list[str]) -> bool:
    """Return True when evidence text contains likely negation markers."""
    combined = " ".join(evidence_texts).lower()
    for kw in NEGATION_KEYWORDS:
        if kw in combined:
            return True
    for mod in NEGATION_TRIGGER_MODIFIERS:
        if mod in combined:
            idx = combined.index(mod)
            snippet = combined[idx : idx + 300]
            if any(kw in snippet for kw in NEGATION_KEYWORDS):
                return True
    return False
