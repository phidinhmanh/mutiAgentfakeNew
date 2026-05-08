"""Identity and numeric discrepancy detection for fact-checking."""

import re
from typing import Any


def extract_person_names(text: str) -> set[str]:
    """Extract likely person names from Vietnamese claim/evidence text."""
    names: set[str] = set()
    for match in re.findall(r"\bA\s+[A-ZÀ-Ỵ][\wÀ-ỵ]*\b", text):
        names.add(match.strip())
    for match in re.findall(
        r"\b(?:anh|chị|cháu|em|ông|bà|bác|cô)\s+([A-ZÀ-Ỵ][\wÀ-ỵ]*(?:\s+[A-ZÀ-Ỵ][\wÀ-ỵ]*){0,2})",
        text,
    ):
        names.add(match.strip())
    return names


def detect_identity_mismatch(claim: str, evidence: list[dict[str, Any]]) -> tuple[bool, str | None]:
    """Compare person-name entities directly between claim and evidence."""
    claim_names = extract_person_names(claim)
    if not claim_names:
        return False, None
    evidence_names: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        text = str(item.get("content", item.get("text", "")))
        evidence_names.update(extract_person_names(text))
    if not evidence_names:
        return False, None
    if claim_names.isdisjoint(evidence_names):
        claim_text = ", ".join(sorted(claim_names))
        evidence_text = ", ".join(sorted(list(evidence_names)[:3]))
        return True, (
            f"Phát hiện sai lệch thực thể tên người ({claim_text} vs {evidence_text}) trong bằng chứng."
        )
    return False, None


def extract_quantitative_numbers(text: str) -> list[float]:
    """Extract quantitative numbers from Vietnamese text, excluding standalone years."""
    values: list[float] = []
    for raw in re.findall(r"\b\d+(?:[.,]\d+)*\b", text):
        num_val = float(raw.replace(".", "").replace(",", "."))
        if 1000 <= num_val <= 2100 and len(raw) == 4:
            continue
        values.append(num_val)
    return values


def detect_numeric_discrepancy(
    claim: str,
    evidence: list[dict[str, Any]],
    threshold: float = 0.01,
    verifier_reasoning: str | None = None,
) -> tuple[bool, str | None]:
    """Compare claim vs evidence numbers directly for high-precision mismatches."""
    claim_numbers = [n for n in extract_quantitative_numbers(claim) if n >= 100]
    if not claim_numbers:
        return False, None
    evidence_numbers: list[float] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        text = str(item.get("content", item.get("text", "")))
        evidence_numbers.extend(n for n in extract_quantitative_numbers(text) if n >= 100)
    reasoning_texts = [verifier_reasoning] if verifier_reasoning else []
    for item in evidence:
        if isinstance(item, dict):
            reasoning_texts.append(str(item.get("content", "")))
            reasoning_texts.append(str(item.get("text", "")))
    all_reasoning = " ".join(filter(None, reasoning_texts))
    reasoning_numbers = extract_quantitative_numbers(all_reasoning)
    numbers_to_check = list(set(evidence_numbers + reasoning_numbers))
    if not numbers_to_check:
        return False, None
    for claim_num in claim_numbers:
        closest = min(numbers_to_check, key=lambda n: abs(n - claim_num))
        if closest <= 0:
            continue
        rel_diff = abs(claim_num - closest) / closest
        if rel_diff > threshold:
            reasoning = (
                "Phát hiện sai lệch số liệu định lượng "
                f"({closest:,.0f} vs {claim_num:,.0f}) vượt ngưỡng 1%."
            )
            return True, reasoning.replace(",", ".")
    return False, None