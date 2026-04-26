"""
Logic Aggregator Agent - Reconstructs truth from atomic claim verdicts.

Uses logical structure to combine atomic claim verdicts into final verdict.
Based on: LoCal (ACM 2024)
"""

import logging
import re
from typing import Any

logger = logging.getLogger("TRUST_agents.agents.logic_aggregator")

TruthValue = bool | None


class _LogicParseError(ValueError):
    pass


class LogicAggregator:
    """
    Aggregates atomic claim verdicts using logical structure.

    Supports: AND, OR, NOT, IMPLIES
    """

    def __init__(self) -> None:
        self.truth_map = {
            "true": True,
            "supported": True,
            "false": False,
            "contradicted": False,
            "refuted": False,
            "uncertain": None,
        }

    def aggregate(
        self,
        atomic_verdicts: list[dict[str, Any]],
        logic_structure: str,
    ) -> dict[str, Any]:
        """
        Aggregate atomic verdicts using logical structure.

        Args:
            atomic_verdicts: List of verdicts for atomic claims
            logic_structure: Logical formula (e.g., "C1 AND C2")

        Returns:
            Final verdict with reasoning
        """
        logger.info("[LOGIC_AGG] Aggregating %d verdicts", len(atomic_verdicts))
        logger.info("[LOGIC_AGG] Logic structure: %s", logic_structure)

        truth_values = self._normalize_truth_values(atomic_verdicts)
        logger.debug("[LOGIC_AGG] Truth values: %s", truth_values)

        try:
            result = self._evaluate_logic(logic_structure, truth_values)

            if result is True:
                final_verdict = "true"
                confidence = self._compute_confidence(atomic_verdicts, "true")
            elif result is False:
                final_verdict = "false"
                confidence = self._compute_confidence(atomic_verdicts, "false")
            else:
                final_verdict = "uncertain"
                confidence = 0.3

            reasoning = self._generate_reasoning(
                logic_structure,
                atomic_verdicts,
                final_verdict,
            )

            logger.info(
                "[LOGIC_AGG] Final verdict: %s (%.2f)",
                final_verdict,
                confidence,
            )

            return {
                "verdict": final_verdict,
                "confidence": confidence,
                "label": final_verdict,
                "reasoning": reasoning,
                "logic_structure": logic_structure,
                "atomic_verdicts": atomic_verdicts,
            }
        except Exception as error:
            logger.error("[LOGIC_AGG] Error evaluating logic: %s", error)
            return self._majority_vote_fallback(atomic_verdicts)

    def _normalize_truth_values(
        self,
        atomic_verdicts: list[dict[str, Any]],
    ) -> dict[str, TruthValue]:
        return {
            f"C{i}": self.truth_map.get(
                str(verdict.get("verdict", "uncertain")).lower(),
                None,
            )
            for i, verdict in enumerate(atomic_verdicts, 1)
        }

    def _evaluate_logic(
        self,
        formula: str,
        truth_values: dict[str, TruthValue],
    ) -> TruthValue:
        tokens = self._tokenize(formula)
        value, position = self._parse_implies(tokens, 0, truth_values)
        if position != len(tokens):
            raise _LogicParseError(f"Unexpected token: {tokens[position]}")
        return value

    def _tokenize(self, formula: str) -> list[str]:
        return re.findall(r"C\d+|AND|OR|NOT|IMPLIES|\(|\)", formula.upper())

    def _parse_implies(
        self,
        tokens: list[str],
        position: int,
        truth_values: dict[str, TruthValue],
    ) -> tuple[TruthValue, int]:
        left, position = self._parse_or(tokens, position, truth_values)
        while position < len(tokens) and tokens[position] == "IMPLIES":
            position += 1
            right, position = self._parse_or(tokens, position, truth_values)
            left = self._imply_truth_value(left, right)
        return left, position

    def _parse_or(
        self,
        tokens: list[str],
        position: int,
        truth_values: dict[str, TruthValue],
    ) -> tuple[TruthValue, int]:
        left, position = self._parse_and(tokens, position, truth_values)
        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            right, position = self._parse_and(tokens, position, truth_values)
            left = self._combine_truth_values(left, right, "OR")
        return left, position

    def _parse_and(
        self,
        tokens: list[str],
        position: int,
        truth_values: dict[str, TruthValue],
    ) -> tuple[TruthValue, int]:
        left, position = self._parse_not(tokens, position, truth_values)
        while position < len(tokens) and tokens[position] == "AND":
            position += 1
            right, position = self._parse_not(tokens, position, truth_values)
            left = self._combine_truth_values(left, right, "AND")
        return left, position

    def _parse_not(
        self,
        tokens: list[str],
        position: int,
        truth_values: dict[str, TruthValue],
    ) -> tuple[TruthValue, int]:
        if position < len(tokens) and tokens[position] == "NOT":
            value, position = self._parse_not(tokens, position + 1, truth_values)
            return self._negate_truth_value(value), position
        return self._parse_primary(tokens, position, truth_values)

    def _parse_primary(
        self,
        tokens: list[str],
        position: int,
        truth_values: dict[str, TruthValue],
    ) -> tuple[TruthValue, int]:
        if position >= len(tokens):
            raise _LogicParseError("Unexpected end of formula")

        token = tokens[position]
        if token == "(":
            value, position = self._parse_implies(tokens, position + 1, truth_values)
            if position >= len(tokens) or tokens[position] != ")":
                raise _LogicParseError("Missing closing parenthesis")
            return value, position + 1

        if token not in truth_values:
            raise _LogicParseError(f"Unknown token: {token}")
        return truth_values[token], position + 1

    def _combine_truth_values(
        self,
        left: TruthValue,
        right: TruthValue,
        operator: str,
    ) -> TruthValue:
        if left is None or right is None:
            return None
        if operator == "AND":
            return left and right
        if operator == "OR":
            return left or right
        raise _LogicParseError(f"Unsupported operator: {operator}")

    def _negate_truth_value(self, value: TruthValue) -> TruthValue:
        if value is None:
            return None
        return not value

    def _imply_truth_value(self, left: TruthValue, right: TruthValue) -> TruthValue:
        if left is None or right is None:
            return None
        return (not left) or right

    def _compute_confidence(
        self,
        atomic_verdicts: list[dict[str, Any]],
        final_verdict: str,
    ) -> float:
        agreeing_confidences = []

        for verdict in atomic_verdicts:
            verdict_label = str(verdict.get("verdict", "uncertain")).lower()
            confidence = float(verdict.get("confidence", 0.5))

            if (final_verdict == "true" and verdict_label in ["true", "supported"]) or (
                final_verdict == "false" and verdict_label in ["false", "contradicted"]
            ):
                agreeing_confidences.append(confidence)

        if agreeing_confidences:
            return sum(agreeing_confidences) / len(agreeing_confidences)
        return 0.3

    def _generate_reasoning(
        self,
        logic_structure: str,
        atomic_verdicts: list[dict[str, Any]],
        final_verdict: str,
    ) -> str:
        reasoning_parts = []

        for i, verdict in enumerate(atomic_verdicts, 1):
            claim = verdict.get("claim", f"Claim {i}")
            verdict_label = verdict.get("verdict", "uncertain")
            confidence = verdict.get("confidence", 0.5)
            reasoning_parts.append(
                f"C{i} ({str(claim)[:50]}...): {verdict_label} ({confidence:.2f})"
            )

        reasoning_parts.append(f"\nLogical structure: {logic_structure}")
        reasoning_parts.append(f"Final verdict: {final_verdict}")
        return "\n".join(reasoning_parts)

    def _majority_vote_fallback(
        self,
        atomic_verdicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        logger.warning("[LOGIC_AGG] Using majority vote fallback")

        votes = {"true": 0, "false": 0, "uncertain": 0}
        for verdict in atomic_verdicts:
            verdict_label = str(verdict.get("verdict", "uncertain")).lower()
            if verdict_label in ["true", "supported"]:
                votes["true"] += 1
            elif verdict_label in ["false", "contradicted"]:
                votes["false"] += 1
            else:
                votes["uncertain"] += 1

        final_verdict = max(votes, key=votes.get)
        confidence = (
            votes[final_verdict] / len(atomic_verdicts) if atomic_verdicts else 0.0
        )

        return {
            "verdict": final_verdict,
            "confidence": confidence,
            "label": final_verdict,
            "reasoning": f"Majority vote: {votes}",
            "fallback": True,
        }


if __name__ == "__main__":
    aggregator = LogicAggregator()

    verdicts = [
        {"claim": "Biden won 2020 election", "verdict": "true", "confidence": 0.9},
        {
            "claim": "Biden became president in 2021",
            "verdict": "true",
            "confidence": 0.95,
        },
    ]
    print(aggregator.aggregate(verdicts, "C1 AND C2"))

    verdicts = [
        {"claim": "Claim A", "verdict": "false", "confidence": 0.8},
        {"claim": "Claim B", "verdict": "true", "confidence": 0.7},
    ]
    print(aggregator.aggregate(verdicts, "C1 OR C2"))

    verdicts = [
        {"claim": "Condition", "verdict": "true", "confidence": 0.9},
        {"claim": "Consequence", "verdict": "false", "confidence": 0.8},
    ]
    print(aggregator.aggregate(verdicts, "C1 IMPLIES C2"))
