"""
TRUST Agents Orchestrator - Coordinates all agents in the fact-checking pipeline.

Pipeline:
1. Claim Extractor: Extract factual claims from text
2. Evidence Retriever: Find relevant evidence for each claim
3. Verifier: Verify claims against evidence
4. Explainer: Generate comprehensive explanations

This orchestrator runs the complete end-to-end pipeline.

FIXED: Normalizes verifier output to ensure consistent verdict format and confidence range
"""

import concurrent.futures
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv

# Import all agents - use relative imports for the new structure
from trust_agents.agents.claim_extractor import run_claim_extractor_agent_sync
from trust_agents.agents.evidence_retrieval import run_evidence_retrieval_agent_sync
from trust_agents.agents.explainer import run_explainer_agent_sync
from trust_agents.agents.verifier import run_verifier_agent_sync

load_dotenv()
logger = logging.getLogger("TRUST_agents.orchestrator")


# ---------------------------------------------------------------------------
# Semantic Cache - reuse results for similar claims (same entities/dates)
# ---------------------------------------------------------------------------
_claim_cache: dict[str, tuple[dict[str, Any], float]] = {}
_CACHE_TTL = 600  # 10 minutes


def _get_claim_signature(claim: str) -> str:
    """Extract stable signature from claim for cache lookup.

    Normalizes claim by:
    - Removing diacritics for consistent matching
    - Extracting key entities (names, dates, numbers)
    - Sorting for consistent cache key

    IMPORTANT: Signature must be specific enough to avoid cross-contamination.
    Different claims about different people/events must have different signatures.
    """
    import unicodedata

    # Normalize diacritics
    nfd = unicodedata.normalize("NFD", claim.lower())
    text = "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    # Extract key features: proper nouns, dates, large numbers
    features = set()

    # Years (4-digit)
    features.update(re.findall(r"\b20\d{2}\b", text))
    # Dates (dd/mm or similar)
    features.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text))
    # Large numbers (3+ digits)
    features.update(re.findall(r"\b\d{3,}\b", text))

    # Proper nouns (capitalized words, excluding stopwords)
    # CRITICAL: Include person names to avoid cross-contamination
    stopwords = {
        "thủ", "tướng", "bộ", "trưởng", "chủ", "tịch", "ông", "bà",
        "việt", "nam", "hà", "nội", "tp", "hcm",
    }
    for word in re.findall(r"\b[a-zà-ỹ]{3,}\b", text):
        if word not in stopwords:
            features.add(word)

    # Add first 3 words of claim for additional specificity
    # This prevents "EXO ra mắt 2012" from matching "Mbappe ghi bàn 2012"
    first_words = text.split()[:3]
    features.update(w for w in first_words if len(w) >= 3)

    # Sort for consistent key
    sig = "|".join(sorted(features))
    # Hash if too long
    if len(sig) > 100:
        sig = hashlib.md5(sig.encode()).hexdigest()

    return sig


def _get_cached_result(claim: str) -> dict[str, Any] | None:
    """Get cached result for claim if fresh."""
    sig = _get_claim_signature(claim)
    if sig in _claim_cache:
        result, cached_at = _claim_cache[sig]
        import time
        if time.time() - cached_at < _CACHE_TTL:
            logger.info(f"[CACHE] HIT for claim: {claim[:50]}... (sig={sig[:16]}...)")
            return result.copy()
    return None


def _cache_result(claim: str, result: dict[str, Any]) -> None:
    """Cache result for claim."""
    import time
    sig = _get_claim_signature(claim)
    _claim_cache[sig] = (result.copy(), time.time())
    logger.info(f"[CACHE] Stored result for sig={sig[:16]}... ({len(_claim_cache)} entries)")


def clear_claim_cache() -> None:
    """Clear semantic claim cache."""
    global _claim_cache
    _claim_cache.clear()
    logger.info("[CACHE] Claim cache cleared")


@dataclass
class TRUSTResult:
    """Result from complete TRUST Agents pipeline."""

    original_text: str
    claims: list[str]
    results: list[dict[str, Any]]
    summary: dict[str, Any]


class TRUSTOrchestrator:
    """
    Orchestrator for TRUST Agents multi-agent system.

    Coordinates:
    - Claim Extractor
    - Evidence Retriever
    - Verifier
    - Explainer
    """

    def __init__(
        self,
        index_dir: str = "retrieval_index",
        top_k_evidence: int = 5,
        max_claim_workers: int = 2,
        ground_truth_evidence: str | None = None,
        use_gt_fallback: bool = True,
    ):
        """
        Initialize TRUST orchestrator.

        Args:
            index_dir: Directory for retrieval index
            top_k_evidence: Number of evidence passages to retrieve per claim
            max_claim_workers: Max concurrent claim workers
            ground_truth_evidence: Ground truth evidence text for benchmark fallback
            use_gt_fallback: Use ground truth if all search providers fail
        """
        self.index_dir = index_dir
        self.top_k_evidence = top_k_evidence
        self.max_claim_workers = max(1, max_claim_workers)
        self.ground_truth_evidence = ground_truth_evidence
        self.use_gt_fallback = use_gt_fallback
        logger.info("[ORCHESTRATOR] TRUST Agents initialized")
        logger.info("[ORCHESTRATOR] Claim worker concurrency: %d", self.max_claim_workers)
        if ground_truth_evidence:
            logger.info(
                "[ORCHESTRATOR] Ground truth fallback enabled (%d chars)",
                len(ground_truth_evidence),
            )

    def _normalize_verdict(self, verdict_data: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize verifier output to ensure consistent format.

        Fixes:
        - Maps "supported" -> "true", "contradicted" -> "false"
        - Extracts verdict from text like "The claim is uncertain" -> "uncertain"
        - Ensures confidence is 0.0-1.0 (not 0-100)
        - Handles errors and edge cases

        Args:
            verdict_data: Raw output from verifier

        Returns:
            Normalized verdict data
        """
        # Make a copy to avoid modifying original
        normalized = verdict_data.copy()

        # Extract verdict
        raw_verdict = verdict_data.get("verdict", "uncertain")

        # Clean up verdict text
        if isinstance(raw_verdict, str):
            verdict_lower = raw_verdict.lower().strip()

            # Map verifier outputs to standard labels
            verdict_map = {
                "supported": "true",
                "true": "true",
                "contradicted": "false",
                "false": "false",
                "insufficient": "uncertain",
                "uncertain": "uncertain",
                "error": "uncertain",
            }

            # Try direct mapping first
            if verdict_lower in verdict_map:
                normalized["verdict"] = verdict_map[verdict_lower]
                normalized["label"] = verdict_map[verdict_lower]
            else:
                # Extract from text like "The claim is false" or "The verdict is uncertain"
                if "false" in verdict_lower or "contradicted" in verdict_lower:
                    normalized["verdict"] = "false"
                    normalized["label"] = "false"
                elif "true" in verdict_lower or "supported" in verdict_lower:
                    normalized["verdict"] = "true"
                    normalized["label"] = "true"
                else:
                    # Default to uncertain for anything else
                    normalized["verdict"] = "uncertain"
                    normalized["label"] = "uncertain"
                    logger.warning(f"[ORCHESTRATOR] Unusual verdict format: {raw_verdict}, defaulting to uncertain")
        else:
            # Non-string verdict
            logger.warning(f"[ORCHESTRATOR] Non-string verdict: {type(raw_verdict)}, defaulting to uncertain")
            normalized["verdict"] = "uncertain"
            normalized["label"] = "uncertain"

        # Normalize confidence to 0.0-1.0
        raw_confidence = verdict_data.get("confidence", 0.3)

        try:
            if isinstance(raw_confidence, str):
                # Try to parse string
                confidence = float(raw_confidence)
            else:
                confidence = float(raw_confidence)

            # If confidence is > 1, assume it's in 0-100 format
            if confidence > 1.0:
                confidence = confidence / 100.0
                logger.debug(f"[ORCHESTRATOR] Converted confidence from percentage: {raw_confidence} -> {confidence}")

            # Clamp to valid range
            confidence = max(0.0, min(1.0, confidence))

        except (ValueError, TypeError):
            logger.warning(f"[ORCHESTRATOR] Could not parse confidence: {raw_confidence}, defaulting to 0.3")
            confidence = 0.3

        normalized["confidence"] = float(confidence)

        # Ensure reasoning exists
        if "reasoning" not in normalized or not normalized["reasoning"]:
            normalized["reasoning"] = normalized.get("evidence_summary", {}).get("reasoning", "No reasoning provided")

        return normalized

    def process_text(self, text: str, skip_evidence: bool = False) -> TRUSTResult:
        """
        Process text through complete TRUST pipeline.

        Args:
            text: Input text to fact-check
            skip_evidence: If True, skip evidence retrieval (for testing)

        Returns:
            TRUSTResult with complete analysis
        """
        logger.info("[ORCHESTRATOR] Starting TRUST pipeline")
        logger.info("[ORCHESTRATOR] Input text length: %d characters", len(text))

        # Step 1: Extract Claims
        logger.info("[ORCHESTRATOR] STEP 1: Extracting claims...")
        claims = run_claim_extractor_agent_sync(text)
        logger.info("[ORCHESTRATOR] Extracted %d claims", len(claims))

        if not claims:
            logger.warning("[ORCHESTRATOR] No claims extracted, stopping pipeline")
            return TRUSTResult(
                original_text=text,
                claims=[],
                results=[],
                summary={"status": "no_claims", "message": "No claims found in text"},
            )

        # Process each claim
        results = self._process_claims(claims, skip_evidence)

        # Create summary
        summary = self._create_summary(results)

        logger.info("[ORCHESTRATOR] Pipeline complete. Processed %d claims", len(results))

        return TRUSTResult(original_text=text, claims=claims, results=results, summary=summary)

    def _process_claims(self, claims: list[str], skip_evidence: bool = False) -> list[dict[str, Any]]:
        """Process claims with bounded concurrency while preserving order."""
        if len(claims) <= 1 or self.max_claim_workers == 1:
            return [
                self._process_single_claim_with_fallback(index, claim, len(claims), skip_evidence)
                for index, claim in enumerate(claims, 1)
            ]

        max_workers = min(self.max_claim_workers, len(claims))
        logger.info(
            "[ORCHESTRATOR] Processing %d claims with %d workers",
            len(claims),
            max_workers,
        )

        indexed_results: list[dict[str, Any] | None] = [None] * len(claims)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_single_claim_with_fallback, i, claim, len(claims), skip_evidence): i - 1
                for i, claim in enumerate(claims, 1)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    indexed_results[idx] = future.result()
                except Exception as e:
                    logger.error("Worker for claim %d failed: %s", idx + 1, e)

        return [result for result in indexed_results if result is not None]

    def _process_single_claim_with_fallback(
        self,
        index: int,
        claim: str,
        total_claims: int,
        skip_evidence: bool = False,
    ) -> dict[str, Any]:
        """Process one claim with retry logic for transient errors.

        Uses semantic cache to reuse results for similar claims.
        """
        logger.info(
            "[ORCHESTRATOR] Processing claim %d/%d: %s",
            index,
            total_claims,
            claim[:80],
        )

        # Check semantic cache first
        cached = _get_cached_result(claim)
        if cached is not None:
            logger.info(f"[ORCHESTRATOR] Using cached result for claim {index}")
            cached["claim"] = claim  # Ensure claim is set
            return cached

        # Retry logic for transient errors (502, 429, 500, 503, etc.)
        max_retries = 5
        base_delay = 30  # Increased for stability
        last_error = None

        for attempt in range(max_retries):
            try:
                result = self._process_single_claim(claim, skip_evidence)
                logger.info(
                    "[ORCHESTRATOR] Claim %d complete: verdict=%s",
                    index,
                    result.get("verdict"),
                )
                # Store in semantic cache
                _cache_result(claim, result)
                return result
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                is_transient = any(
                    code in error_str
                    for code in [
                        "502",
                        "429",
                        "500",
                        "503",
                        "504",
                        "bad gateway",
                        "rate_limit",
                        "timeout",
                        "read timed out",
                        "connection error",
                    ]
                )

                if is_transient and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "[ORCHESTRATOR] Claim %d attempt %d failed (transient: %s). Retrying in %ds...",
                        index,
                        attempt + 1,
                        str(e)[:50],
                        delay,
                    )
                    import time

                    time.sleep(delay)
                else:
                    # Final failure after all retries
                    logger.error(
                        "[ORCHESTRATOR] Error processing claim %d after %d attempts: %s",
                        index,
                        attempt + 1,
                        e,
                        exc_info=True,
                    )
                    return {
                        "claim": claim,
                        "verdict": "uncertain",
                        "confidence": 0.0,
                        "label": "uncertain",
                        "reasoning": f"Error after {attempt + 1} attempts: {str(e)}",
                        "error": str(e),
                    }

        return {
            "claim": claim,
            "verdict": "uncertain",
            "confidence": 0.0,
            "label": "uncertain",
            "reasoning": f"Failed after {max_retries} attempts. Last error: {str(last_error)}",
        }

    def _process_single_claim(self, claim: str, skip_evidence: bool = False) -> dict[str, Any]:
        """
        Process a single claim through the pipeline.

        Note: Retry logic is handled in _process_single_claim_with_fallback.
        """
        # Step 2: Retrieve Evidence
        if skip_evidence:
            logger.info("[ORCHESTRATOR] Skipping evidence retrieval")
            evidence = []
        else:
            logger.info("[ORCHESTRATOR] STEP 2: Retrieving evidence...")
            evidence = run_evidence_retrieval_agent_sync(
                claim,
                top_k=self.top_k_evidence,
                ground_truth_evidence=self.ground_truth_evidence,
                use_gt_fallback=self.use_gt_fallback,
            )
            logger.info("[ORCHESTRATOR] Retrieved %d evidence passages", len(evidence))

        # Step 3: Verify Claim
        logger.info("[ORCHESTRATOR] STEP 3: Verifying claim...")
        if evidence:
            verdict_data = run_verifier_agent_sync(claim, evidence)
        else:
            # No evidence available
            verdict_data = {
                "claim": claim,
                "verdict": "uncertain",
                "confidence": 0.1,
                "label": "uncertain",
                "reasoning": "No evidence available for verification",
            }

        # Normalize the verdict output
        verdict_data = self._normalize_verdict(verdict_data)

        logger.info(
            "[ORCHESTRATOR] Verification complete: %s (%.1f%%)",
            verdict_data.get("verdict"),
            verdict_data.get("confidence", 0) * 100,
        )

        # Step 4: Generate Explanation
        logger.info("[ORCHESTRATOR] STEP 4: Generating explanation...")
        report = run_explainer_agent_sync(claim, verdict_data, evidence)

        # Ensure report has normalized verdict
        if "verdict" in report:
            report = self._normalize_verdict(report)
        else:
            # Merge verdict data into report if not present
            report.update(
                {
                    "verdict": verdict_data["verdict"],
                    "confidence": verdict_data["confidence"],
                    "label": verdict_data["label"],
                }
            )

        logger.info("[ORCHESTRATOR] Explanation complete")
        return report

    def _create_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Create summary statistics for all results using Aggregator V4.

        V4 changes from V3:
        - Source authority weighting: .gov.vn, .chinhphu.vn, major news get 2x weight
        - Weighted average confidence instead of simple average
        - Source diversity scoring for final verdict confidence
        """
        if not results:
            return {"total_claims": 0}

        # ── Source authority scoring ───────────────────────────────────────────
        # High-authority domains get 2x weight
        high_authority_domains = {
            ".gov.vn", ".chinhphu.vn", "baochinhphu.vn",
            "vnexpress.net", "tuoitre.vn", "thanhnien.vn",
            "nhandan.vn", "vietnamplus.vn", "laodong.vn",
            "dantri.com.vn", "nld.com.vn",
        }
        gov_domains = {".gov.vn", ".chinhphu.vn", "baochinhphu.vn"}

        def _get_source_authority(evidence: list[dict[str, Any]]) -> float:
            """Calculate authority weight for evidence sources."""
            if not evidence:
                return 1.0  # default weight

            max_weight = 1.0
            for item in evidence:
                url = item.get("url", "")
                source = item.get("source", "")
                domain = url.lower() if url else source.lower()

                # Check for high-authority domains
                for auth_domain in high_authority_domains:
                    if auth_domain in domain:
                        max_weight = max(max_weight, 2.0)
                        if auth_domain in gov_domains:
                            max_weight = max(max_weight, 2.5)
                        break

            return max_weight

        # Count verdicts with weights
        verdict_counts = {"true": 0, "false": 0, "uncertain": 0, "error": 0}
        weighted_confidences = []
        source_authorities = []

        for result in results:
            verdict = result.get("verdict", "uncertain")

            # Normalize verdict for counting
            if verdict in ["true", "supported"]:
                verdict_counts["true"] += 1
            elif verdict in ["false", "contradicted"]:
                verdict_counts["false"] += 1
            elif verdict in ["uncertain", "insufficient"]:
                verdict_counts["uncertain"] += 1
            else:
                verdict_counts["error"] += 1

            conf = result.get("confidence", 0.0)
            try:
                conf = float(conf)
                if conf > 1.0:
                    conf = conf / 100.0
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                conf = 0.0

            # Get source authority for this claim
            evidence = result.get("evidence", [])
            authority = _get_source_authority(evidence) if evidence else 1.0
            source_authorities.append(authority)

            # Store weighted confidence
            weighted_confidences.append((conf, authority))

        # ── V4: Weighted average confidence ───────────────────────────────────
        # Instead of simple average, weight by source authority
        total_weight = sum(source_authorities)
        weighted_avg_conf = (
            sum(c * w for c, w in weighted_confidences) / total_weight
            if total_weight > 0 else 0.0
        )
        simple_avg_conf = (
            sum(c for c, _ in weighted_confidences) / len(weighted_confidences)
            if weighted_confidences else 0.0
        )

        # Map: true→REAL, false→FAKE, uncertain→UNCERTAIN (for counting)
        norm = {"true": "REAL", "false": "FAKE", "uncertain": "UNCERTAIN"}
        norm_verdicts = [norm.get(c.get("verdict", "uncertain"), "UNCERTAIN") for c in results]
        norm_verdicts = [v for v in norm_verdicts if v in ("REAL", "FAKE", "UNCERTAIN")]

        total = len(norm_verdicts)
        uncertain_count = norm_verdicts.count("UNCERTAIN")

        # ── V4: Weighted verdict scoring ─────────────────────────────────────
        real_weight = 0.0
        fake_weight = 0.0
        uncertain_weight = 0.0

        for i, result in enumerate(results):
            verdict = result.get("verdict", "uncertain")
            authority = source_authorities[i] if i < len(source_authorities) else 1.0

            if verdict in ["true", "supported"]:
                real_weight += authority
            elif verdict in ["false", "contradicted"]:
                fake_weight += authority
            elif verdict in ["uncertain", "insufficient"]:
                uncertain_weight += authority

        # ── Final verdict logic V4 ───────────────────────────────────────────
        if total == 0:
            final_verdict = "UNCERTAIN"
            final_confidence = 0.0
        elif fake_weight >= 3.0:
            final_verdict = "FAKE"
            final_confidence = weighted_avg_conf
        elif fake_weight >= 2.0 and weighted_avg_conf >= 0.80:
            final_verdict = "FAKE"
            final_confidence = weighted_avg_conf
        elif real_weight >= 2.0 and fake_weight < 1.0:
            final_verdict = "REAL"
            final_confidence = weighted_avg_conf
        elif real_weight > fake_weight + 1.0:
            final_verdict = "REAL"
            final_confidence = weighted_avg_conf
        elif fake_weight > real_weight:
            final_verdict = "FAKE"
            final_confidence = weighted_avg_conf
        elif uncertain_count == total:
            final_verdict = "UNCERTAIN"
            final_confidence = simple_avg_conf * 0.5
        else:
            final_verdict = "UNCERTAIN"
            final_confidence = weighted_avg_conf

        return {
            "verdict": final_verdict,
            "confidence": round(final_confidence, 3),
            "explanation": (
                f"Verified {total} claims with weighted scoring. "
                f"REAL weight: {real_weight:.1f}, FAKE weight: {fake_weight:.1f}"
                if len(results) > 1
                else (results[0].get("reasoning", "") if results else "No claims found")
            ),
            "claims": results,
            "total_claims": len(results),
            "verdicts": verdict_counts,
            "average_confidence": round(simple_avg_conf, 3),
            "weighted_average_confidence": round(weighted_avg_conf, 3),
            "weighted_scores": {
                "real_weight": round(real_weight, 2),
                "fake_weight": round(fake_weight, 2),
                "uncertain_weight": round(uncertain_weight, 2),
            },
        }

def run_trust_pipeline_sync(
    text: str,
    top_k_evidence: int = 5,
    skip_evidence: bool = False,
    ground_truth_evidence: str | None = None,
    use_gt_fallback: bool = True,
) -> dict[str, Any]:
    """
    Run complete TRUST pipeline on text.

    Args:
        text: Input text to fact-check
        top_k_evidence: Number of evidence passages per claim
        skip_evidence: If True, skip evidence retrieval (for testing)
        ground_truth_evidence: Ground truth evidence text for benchmark fallback
        use_gt_fallback: Use ground truth if all search providers fail

    Returns:
        Dictionary with complete results
    """
    orchestrator = TRUSTOrchestrator(
        top_k_evidence=top_k_evidence,
        ground_truth_evidence=ground_truth_evidence,
        use_gt_fallback=use_gt_fallback,
    )
    result = orchestrator.process_text(text, skip_evidence=skip_evidence)
    return asdict(result)


# Convenience function
def fact_check(text: str, top_k_evidence: int = 5) -> dict[str, Any]:
    """
    Fact-check text using TRUST Agents.

    Simple interface for fact-checking.

    Args:
        text: Text to fact-check
        top_k_evidence: Number of evidence passages per claim

    Returns:
        Dictionary with fact-check results
    """
    return run_trust_pipeline_sync(text, top_k_evidence=top_k_evidence)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run TRUST Agents fact-checking pipeline")
    parser.add_argument("--text", required=True, help="Text to fact-check")
    parser.add_argument("--top-k", type=int, default=5, help="Evidence passages per claim")
    parser.add_argument("--skip-evidence", action="store_true", help="Skip evidence retrieval")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # Run pipeline
    result = run_trust_pipeline_sync(args.text, top_k_evidence=args.top_k, skip_evidence=args.skip_evidence)

    # Print results
    print("\n" + "=" * 70)
    print("TRUST AGENTS - FACT-CHECK RESULTS")
    print("=" * 70)
    print(f"\nOriginal Text: {args.text[:200]}...")
    print(f"\nClaims Found: {len(result['claims'])}")

    for i, claim_result in enumerate(result["results"], 1):
        print(f"\n--- Claim {i} ---")
        print(f"Claim: {claim_result['claim']}")
        print(f"Verdict: {claim_result['verdict']} (confidence: {claim_result.get('confidence', 0):.1%})")
        print(f"Summary: {claim_result.get('summary', 'N/A')}")

    print("\n--- Summary ---")
    summary = result["summary"]
    print(f"Total Claims: {summary['total_claims']}")
    print(f"Verdicts: {summary.get('verdicts', {})}")
    print(f"Average Confidence: {summary.get('average_confidence', 0):.1%}")

    # Save if requested
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {args.output}")
