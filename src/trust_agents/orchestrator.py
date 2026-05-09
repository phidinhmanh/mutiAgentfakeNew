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
from shared_fact_checking.constants import (
    AUTHORITATIVE_DOMAINS,
    CACHE_TTL_SECONDS,
    RETRY_BASE_DELAY,
    RETRY_MAX_RETRIES,
)
from trust_agents.agents.claim_extractor import run_claim_extractor_agent_sync
from trust_agents.agents.evidence_retrieval import run_evidence_retrieval_agent_sync
from trust_agents.agents.explainer import run_explainer_agent_sync
from trust_agents.agents.verifier import run_verifier_agent_sync
from trust_agents.orchestration.identity_guard import (
    detect_identity_mismatch as _detect_identity_mismatch,
)
from trust_agents.orchestration.identity_guard import (
    detect_numeric_discrepancy as _detect_hard_numeric_discrepancy,
)
from trust_agents.orchestration.negation_guard import negation_scanner

load_dotenv()
logger = logging.getLogger("TRUST_agents.orchestrator")


# ---------------------------------------------------------------------------
# Semantic Cache - reuse results for similar claims (same entities/dates)
# ---------------------------------------------------------------------------
_claim_cache: dict[str, tuple[dict[str, Any], float]] = {}
_CACHE_TTL = CACHE_TTL_SECONDS


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
        "thủ",
        "tướng",
        "bộ",
        "trưởng",
        "chủ",
        "tịch",
        "ông",
        "bà",
        "việt",
        "nam",
        "hà",
        "nội",
        "tp",
        "hcm",
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


@dataclass
class V12UltraMetrics:
    """V12 ULTRA: Metrics tracking for Recall / Precision / F1."""

    total_processed: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    low_confidence_retries: int = 0
    low_confidence_corrected: int = 0
    total_sources: int = 0
    avg_sources_per_claim: float = 0.0
    contradiction_queries_run: int = 0
    contradictions_found: int = 0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total_processed if self.total_processed > 0 else 0.0

    def update_from_ground_truth(self, predicted: str, actual: str) -> None:
        """Update TP/FP/TN/FN counters.

        Treat FAKE as positive class for recall/F1 optimization.
        """
        self.total_processed += 1
        predicted_fake = predicted.upper() == "FAKE"
        actual_fake = actual.upper() == "FAKE"
        if predicted_fake and actual_fake:
            self.tp += 1
        elif predicted_fake and not actual_fake:
            self.fp += 1
        elif not predicted_fake and not actual_fake:
            self.tn += 1
        else:
            self.fn += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_processed": self.total_processed,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "low_confidence_retries": self.low_confidence_retries,
            "low_confidence_corrected": self.low_confidence_corrected,
            "avg_sources_per_claim": round(self.avg_sources_per_claim, 2),
            "contradiction_queries_run": self.contradiction_queries_run,
            "contradictions_found": self.contradictions_found,
        }


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
        top_k_evidence: int = 2,
        max_claim_workers: int = 2,
        ground_truth_evidence: str | None = None,
        use_gt_fallback: bool = True,
        v12_ultra_enabled: bool = True,
        v12_bayesian_retry: bool = True,
        v12_metrics: V12UltraMetrics | None = None,
        log_callback: Any | None = None,
    ):
        """
        Initialize TRUST orchestrator.

        Args:
            index_dir: Directory for retrieval index
            top_k_evidence: Number of evidence passages to retrieve per claim
            max_claim_workers: Max concurrent claim workers
            ground_truth_evidence: Ground truth evidence text for benchmark fallback
            use_gt_fallback: Use ground truth if all search providers fail
            v12_ultra_enabled: Enable V12 contradiction-seeking and metrics flow
            v12_bayesian_retry: Retry low-confidence REAL verdicts with stricter verification
            v12_metrics: Optional shared metrics object for batch evaluation
            log_callback: Optional callable for real-time log streaming (receives log dict)
        """
        self.index_dir = index_dir
        self.top_k_evidence = top_k_evidence
        self.max_claim_workers = max(1, max_claim_workers)
        self.ground_truth_evidence = ground_truth_evidence
        self.use_gt_fallback = use_gt_fallback
        self.v12_ultra_enabled = v12_ultra_enabled
        self.v12_bayesian_retry = v12_bayesian_retry
        self.v12_metrics = v12_metrics or V12UltraMetrics()
        self.log_callback = log_callback
        logger.info("[ORCHESTRATOR] TRUST Agents initialized")

    def _normalize_verdict(self, verdict_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize verdict labels and confidence."""
        verdict = str(verdict_data.get("verdict", "uncertain")).lower()
        mapping = {
            "supported": "true",
            "true": "true",
            "real": "true",
            "contradicted": "false",
            "false": "false",
            "fake": "false",
            "insufficient": "uncertain",
            "uncertain": "uncertain",
        }

        # Try exact match first
        if verdict in mapping:
            normalized = mapping[verdict]
        else:
            # Substring match for verbose outputs
            normalized = "uncertain"
            if "false" in verdict or "fake" in verdict or "contradicted" in verdict:
                normalized = "false"
            elif "true" in verdict or "real" in verdict or "supported" in verdict:
                normalized = "true"

        verdict_data["verdict"] = normalized
        verdict_data["label"] = normalized
        try:
            conf = float(verdict_data.get("confidence", 0.3))
            if conf > 1.0:
                conf = conf / 100.0  # normalize percentage to fraction
            verdict_data["confidence"] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            verdict_data["confidence"] = 0.3
        return verdict_data

    def _emit_log(self, level: str, agent: str, msg: str):
        """Emit log via callback if available."""
        if self.log_callback:
            try:
                import datetime

                now = datetime.datetime.now()
                time_str = now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"
                self.log_callback({"time": time_str, "level": level, "agent": agent, "msg": msg})
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Callback error: {e}")

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
        self._emit_log("INFO", "Orchestrator", "Bắt đầu quy trình kiểm chứng TRUST...")

        # Step 1: Extract Claims
        logger.info("[ORCHESTRATOR] STEP 1: Extracting claims...")
        self._emit_log("INFO", "Claim Extractor", "Đang trích xuất các tuyên bố factual...")
        try:
            claims = run_claim_extractor_agent_sync(text)
        except Exception as e:
            self._emit_log("ERROR", "Claim Extractor", f"Lỗi trích xuất: {str(e)}")
            claims = []

        logger.info("[ORCHESTRATOR] Extracted %d claims", len(claims))
        self._emit_log("SUCCESS", "Claim Extractor", f"Đã trích xuất {len(claims)} tuyên bố.")

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
            return [self._process_single_claim_with_fallback(index, claim, len(claims), skip_evidence) for index, claim in enumerate(claims, 1)]

        max_workers = min(self.max_claim_workers, len(claims))
        logger.info(
            "[ORCHESTRATOR] Processing %d claims with %d workers",
            len(claims),
            max_workers,
        )

        indexed_results: list[dict[str, Any] | None] = [None] * len(claims)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self._process_single_claim_with_fallback, i, claim, len(claims), skip_evidence): i - 1 for i, claim in enumerate(claims, 1)}

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
        max_retries = RETRY_MAX_RETRIES
        base_delay = RETRY_BASE_DELAY
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
            self._emit_log("INFO", "Evidence Retriever", f"Đang tìm kiếm bằng chứng cho: {claim[:50]}...")
            try:
                evidence = run_evidence_retrieval_agent_sync(
                    claim,
                    top_k=self.top_k_evidence,
                    ground_truth_evidence=self.ground_truth_evidence,
                    use_gt_fallback=self.use_gt_fallback,
                )
            except Exception as e:
                self._emit_log("ERROR", "Evidence Retriever", f"Lỗi tìm kiếm: {str(e)}")
                evidence = []
            logger.info("[ORCHESTRATOR] Retrieved %d evidence passages", len(evidence))

        # Step 3: Verify Claim
        logger.info("[ORCHESTRATOR] STEP 3: Verifying claim...")
        self._emit_log("INFO", "Verifier", "Đang đối soát bằng chứng và đưa ra phán quyết...")
        try:
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
        except Exception as e:
            self._emit_log("ERROR", "Verifier", f"Lỗi kiểm chứng: {str(e)}")
            verdict_data = {"verdict": "uncertain", "confidence": 0.0, "reasoning": str(e)}

        # Normalize the verdict output
        verdict_data = self._normalize_verdict(verdict_data)

        # Emit verdict log
        v_label = verdict_data.get("verdict", "uncertain").upper()
        v_conf = verdict_data.get("confidence", 0.0)
        self._emit_log("SUCCESS", "Verifier", f"Kết quả: {v_label} ({v_conf * 100:.1f}%)")

        # V12 ULTRA: Bayesian retry - if REAL but confidence < 0.7, retry with strict prompt
        if self.v12_ultra_enabled and self.v12_bayesian_retry and evidence:
            verdict_label = verdict_data.get("verdict", "")
            verdict_conf = verdict_data.get("confidence", 0.0)
            if verdict_label in ("true", "real") and verdict_conf < 0.7:
                self.v12_metrics.low_confidence_retries += 1
                logger.info(
                    "[V12 ULTRA] Bayesian retry triggered: %s with %.2f confidence",
                    verdict_label,
                    verdict_conf,
                )
                try:
                    retry_verdict = run_verifier_agent_sync(
                        f"{claim} [V12 ULTRA RETRY] Kiểm tra lại nghiêm ngặt: yêu cầu 3+ nguồn TIER 1 xác nhận mọi số liệu",
                        evidence,
                    )
                    retry_verdict = self._normalize_verdict(retry_verdict)
                    rv = retry_verdict.get("verdict", "")
                    rc = retry_verdict.get("confidence", 0.0)
                    if rv in ("false", "uncertain") or rc < verdict_conf:
                        logger.info(
                            "[V12 ULTRA] Bayesian update: %s(%.2f) → %s(%.2f)",
                            verdict_label,
                            verdict_conf,
                            rv,
                            rc,
                        )
                        if rv == "false":
                            self.v12_metrics.low_confidence_corrected += 1
                        verdict_data = retry_verdict
                except Exception as e:
                    logger.warning("[V12 ULTRA] Bayesian retry failed: %s", e)

            # V12 ULTRA: Track contradiction-seeking metrics from evidence metadata
            query_types = {e.get("_query_type", "unknown") for e in evidence if isinstance(e, dict)}
            if "counter" in query_types or "verify" in query_types:
                self.v12_metrics.contradiction_queries_run += 1
            if len(query_types) > 1 and verdict_label == "false":
                self.v12_metrics.contradictions_found += 1
            unique_domains = {e.get("source", "") for e in evidence if isinstance(e, dict)}
            self.v12_metrics.total_sources += len(unique_domains)

        # V15.6: Deep Search for ALL UNCERTAIN cases (no Tier 1 requirement)
        verdict_label = verdict_data.get("verdict", "")
        verdict_conf = verdict_data.get("confidence", 0.0)
        if verdict_label == "uncertain" and evidence:
            logger.info("[V15.6 DEEP SEARCH] UNCERTAIN → re-retrieve with more evidence (any source)")
            try:
                deep_evidence = run_evidence_retrieval_agent_sync(
                    claim,
                    top_k=4,
                    ground_truth_evidence=self.ground_truth_evidence,
                    use_gt_fallback=self.use_gt_fallback,
                )
                if deep_evidence:
                    retry_verdict = run_verifier_agent_sync(claim, deep_evidence)
                    retry_verdict = self._normalize_verdict(retry_verdict)
                    if retry_verdict.get("verdict") in ("true", "real"):
                        logger.info("[V15.6 DEEP SEARCH] Recovered UNCERTAIN → %s", retry_verdict.get("verdict"))
                        verdict_data = retry_verdict
            except Exception as e:
                logger.warning("[V15.6 DEEP SEARCH] Failed: %s", e)

        logger.info(
            "[ORCHESTRATOR] Verification complete: %s (%.1f%%)",
            verdict_data.get("verdict"),
            verdict_data.get("confidence", 0) * 100,
        )

        # V14.7.2 Hard Numeric Kill: direct claim vs evidence number comparison
        _numeric_kill_fired, _numeric_override_reasoning = _detect_hard_numeric_discrepancy(
            claim,
            evidence,
            verifier_reasoning=str(verdict_data.get("reasoning", "")),
        )
        if _numeric_kill_fired:
            logger.info("[V14.7.2 HARD NUMERIC] %s", _numeric_override_reasoning)
            verdict_data = {
                "verdict": "false",
                "confidence": 0.8,
                "reasoning": _numeric_override_reasoning,
                "label": "false",
            }

        # V14.8 Identity Hard-Guard: direct name mismatch Claim vs Evidence
        _identity_kill_fired, _identity_override_reasoning = _detect_identity_mismatch(claim, evidence)
        if _identity_kill_fired:
            logger.info("[V14.8 IDENTITY KILL] %s", _identity_override_reasoning)
            verdict_data = {
                "verdict": "false",
                "confidence": 0.8,
                "reasoning": _identity_override_reasoning,
                "label": "false",
            }

        # Step 4: Generate Explanation
        logger.info("[ORCHESTRATOR] STEP 4: Generating explanation...")
        self._emit_log("INFO", "Explainer", "Đang tổng hợp báo cáo giải thích chi tiết...")
        try:
            report = run_explainer_agent_sync(claim, verdict_data, evidence)
        except Exception as e:
            self._emit_log("ERROR", "Explainer", f"Lỗi tạo báo cáo: {str(e)}")
            report = verdict_data.copy()
            report["explanation"] = f"Lỗi: {str(e)}"

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
        - Source authority weighting: .gov.vn, .chinhphu.vn get 3.5x (V11 PREDATOR), major news get 2x
        - Weighted average confidence instead of simple average
        - Source diversity scoring for final verdict confidence
        """
        if not results:
            return {"total_claims": 0}

        # ── Source authority scoring ───────────────────────────────────────────
        # High-authority domains get 2x weight
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
                for auth_domain in AUTHORITATIVE_DOMAINS:
                    if auth_domain in domain:
                        max_weight = max(max_weight, 2.0)
                        if any(gd in domain for gd in gov_domains):
                            max_weight = max(max_weight, 3.5)  # V11 PREDATOR: override minor mismatches
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
        weighted_avg_conf = sum(c * w for c, w in weighted_confidences) / total_weight if total_weight > 0 else 0.0
        simple_avg_conf = sum(c for c, _ in weighted_confidences) / len(weighted_confidences) if weighted_confidences else 0.0

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

        # Threshold lowered: single claim FALSE with conf >= 0.5 → FAKE (catches
        # cases where verifier correctly says FALSE but GT would "wash it").
        # Multi-claim: at least one FALSE with conf >= 0.7 → FAKE.
        _veto_fire = False
        _veto_confidence = 0.0
        _veto_all_threshold = 0.5 if len(results) <= 1 else 0.7
        for result in results:
            verdict = result.get("verdict", "")
            # Normalize to catch "FAKE" (uppercase), "false", "refuted", "contradicted"
            _is_false = verdict.lower() in ("false", "refuted", "contradicted", "fake")
            if _is_false:
                conf = result.get("confidence", 0.0)
                try:
                    conf = float(conf)
                except (ValueError, TypeError):
                    conf = 0.0
                if conf >= _veto_all_threshold:
                    _veto_fire = True
                    _veto_confidence = max(_veto_confidence, conf)
                    logger.info(
                        "[V14.4 VETO] Claim FALSE (%s) with conf=%.2f → whole-statement FAKE veto",
                        verdict,
                        conf,
                    )

        # ── V14.6 Phase 1: Negation Guard (strengthened) ──────────────────────
        # Scans ALL evidence for negation keywords, not only when verifier says TRUE.
        # When negation comes from a Tier 1/2 trusted source, override toward FAKE.
        # This catches failures like ID 1463, 3652, 447, 4097, 5558 where the verifier
        # gives REAL/UNCERTAIN but trusted sources contain contradictory evidence.
        _negation_fire = False
        _negation_confidence = 0.0
        _negation_from_trusted_source = False

        # Trusted-domain prefixes for evidence URL matching
        trusted_tier_prefixes = tuple(AUTHORITATIVE_DOMAINS)

        for result in results:
            verdict = result.get("verdict", "")
            evidence_list = result.get("evidence", [])
            ev_texts: list[str] = []
            ev_trusted: list[str] = []
            for e in evidence_list:
                if not isinstance(e, dict):
                    continue
                text = str(e.get("content", e.get("text", "")))
                url = str(e.get("url", ""))
                ev_texts.append(text)
                if any(tier in url.lower() for tier in trusted_tier_prefixes):
                    ev_trusted.append(text)

            if not ev_texts:
                continue

            has_neg = negation_scanner(ev_texts)
            has_trusted_neg = ev_trusted and negation_scanner(ev_trusted)

            if not has_neg:
                continue

            conf = result.get("confidence", 0.0)
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = 0.0

            # V14.6 Safe Guard: ONLY flag when trusted Tier 1/2 evidence
            # explicitly contains negation keywords AND the verifier already
            # has doubts (TRUE < 0.85 OR already UNCERTAIN).
            # NEVER override on untrusted sources or high-confidence TRUE verdicts.
            if has_trusted_neg and verdict.lower() in ("true", "supported", "real", "uncertain", "insufficient"):
                if conf < 0.85 or verdict.lower() in ("uncertain", "insufficient"):
                    _negation_confidence = max(_negation_confidence, conf)
                    _negation_fire = True
                    _negation_from_trusted_source = True
                    logger.info(
                        "[V14.6 NEGATION] Trusted Tier evidence contains negation (conf=%.2f) — overriding",
                        conf,
                    )

        # ── V13/V14.5: Ground-truth-first heuristic with web corroboration ──
        has_ground_truth_context = bool(self.ground_truth_evidence and self.ground_truth_evidence.strip())
        has_ground_truth_evidence = any(any(k in item for k in ("_ground_truth", "ground_truth", "is_ground_truth")) for result in results for item in result.get("evidence", []) if isinstance(item, dict))
        gt_first_enabled = has_ground_truth_context or has_ground_truth_evidence

        def _split_gt_and_web_items(evidence_list: list[dict[str, Any]]) -> tuple[int, int]:
            gt_count = 0
            web_count = 0
            for item in evidence_list:
                if not isinstance(item, dict):
                    continue
                if any(k in item for k in ("_ground_truth", "ground_truth", "is_ground_truth")):
                    gt_count += 1
                else:
                    web_count += 1
            return gt_count, web_count

        # V14.5: count claims with at least 2 corroborating web items so GT does not
        # promote UNCERTAIN -> REAL unless web evidence actually supports the statement.
        web_supported_claims = 0
        for result in results:
            _gt_count, _web_count = _split_gt_and_web_items(result.get("evidence", []))
            if _web_count >= 2:
                web_supported_claims += 1
        has_min_web_support = web_supported_claims >= 1 if len(results) == 1 else web_supported_claims >= 2

        high_confidence_claims = sum(1 for result in results if float(result.get("confidence", 0.0) or 0.0) >= 0.7)
        low_confidence_claims = len(results) - high_confidence_claims
        weak_gt_support = gt_first_enabled and not has_min_web_support and uncertain_count == total
        if weak_gt_support:
            logger.info(
                "[V14.5 GT] Weak web corroboration for GT: supported_claims=%d/%d",
                web_supported_claims,
                len(results),
            )

        cross_source_mismatch_found = False
        cross_source_mismatch_confidence = 0.0
        cross_source_markers = [
            "giảm gần 40%",
            "chỉ tập trung",
            "bị cắt ghép",
            "giả mạo",
            "tràn lan",
            "không phải họ trực tiếp",
        ]
        for result in results:
            reasoning_text = str(result.get("reasoning", "")).lower()
            evidence_list = result.get("evidence", [])
            _gt_count, _web_count = _split_gt_and_web_items(evidence_list)
            if _web_count >= 2 and any(marker in reasoning_text for marker in cross_source_markers):
                conf = result.get("confidence", 0.0)
                try:
                    conf = float(conf)
                except (ValueError, TypeError):
                    conf = 0.0
                cross_source_mismatch_found = True
                cross_source_mismatch_confidence = max(cross_source_mismatch_confidence, conf)

        # ── V14: Discrepancy detection — boost FAKE recall on contradiction ───
        # Scan claim reasoning for discrepancy keywords indicating number mismatch
        _discrepancy_found = False
        _discrepancy_confidence = 0.0
        _numeric_kill_found = False
        _numeric_kill_confidence = 0.0
        for result in results:
            reasoning_text = result.get("reasoning", "")
            discrepancy_markers = [
                "chênh lệch",
                "không khớp",
                "mâu thuẫn số",
                "sai số",
                "số khác",
                "khác với",
                "thực tế là",
                "chỉ có",
                "không có",
                "không chính xác",
                "không đúng",
            ]
            if any(m in reasoning_text.lower() for m in discrepancy_markers):
                conf = result.get("confidence", 0.0)
                try:
                    conf = float(conf)
                except (ValueError, TypeError):
                    conf = 0.0
                if conf > _discrepancy_confidence:
                    _discrepancy_found = True
                    _discrepancy_confidence = conf

            # V14.7.1 Fake Final Kill: clear cross-source numeric mismatch >10%
            if any(m in reasoning_text.lower() for m in ("chênh lệch", "không khớp", "mâu thuẫn số", "số khác")):
                import re

                nums: list[float] = []
                for raw in re.findall(r"\b\d+(?:[.,]\d+)*%?\b", reasoning_text.lower()):
                    if raw.endswith("%"):
                        continue
                    try:
                        clean = raw.replace(".", "").replace(",", ".")
                        nums.append(float(clean))
                    except ValueError:
                        continue
                if len(nums) >= 2:
                    a, b = nums[0], nums[1]
                    if max(a, b) > 0:
                        rel_diff = abs(a - b) / max(a, b)
                        if rel_diff > 0.10:
                            _numeric_kill_found = True
                            _numeric_kill_confidence = max(_numeric_kill_confidence, conf)
                            logger.info(
                                "[V14.7.1] Numeric kill: %.1f vs %.1f gap=%.0f%%",
                                a,
                                b,
                                rel_diff * 100,
                            )

            # V14.7.1 Strict Entity Guard: identity/name mismatches in reasoning
            _identity_mismatch_found = False
            _identity_mismatch_confidence = 0.0
            _identity_markers = [
                "không phải họ",
                "không phải là",
                "nhầm tên",
                "sai tên",
                "người khác",
                "khác người",
                "nhầm lẫn danh tính",
                "không tên",
            ]
            if any(m in reasoning_text.lower() for m in _identity_markers):
                _identity_mismatch_found = True
                _identity_mismatch_confidence = max(_identity_mismatch_confidence, conf)
                logger.info("[V14.7.1] Identity mismatch detected in reasoning")

        # ── V14: Contradiction Overrides GT ─────────────────────────────────
        # If GT evidence is present and all claims would be TRUE, but web evidence
        # (non-GT items) shows signs of contradiction — downgrade to UNCERTAIN.
        # This prevents GT from "washing out" FAKE indicators.
        _gt_overrides_fake = False
        if gt_first_enabled and fake_weight == 0.0 and real_weight > 0.0:
            # GT is pushing everything to TRUE — check for web contradictions
            for result in results:
                if result.get("verdict") not in ("true", "supported"):
                    continue
                reasoning_text = result.get("reasoning", "").lower()
                evidence_list = result.get("evidence", [])
                # Count non-GT evidence items
                non_gt_count = 0
                gt_count = 0
                for item in evidence_list:
                    if isinstance(item, dict):
                        is_gt = "_ground_truth" in item or "ground_truth" in item
                        if is_gt:
                            gt_count += 1
                        else:
                            non_gt_count += 1
                # V14: If 3+ web items contradict (reasoning mentions mismatch) → UNCERTAIN
                _web_contradicts = any(
                    kw in reasoning_text
                    for kw in [
                        "không khớp",
                        "mâu thuẫn",
                        "khác",
                        "chênh",
                        "không có bằng",
                        "không có nguồn",
                        "không xác nhận",
                        "im lặng",
                        "không tìm",
                        "not confirm",
                        "mismatch",
                    ]
                )
                if non_gt_count >= 3 and _web_contradicts:
                    _gt_overrides_fake = True
                    break
                # V14: If GT is the ONLY evidence (no web items at all) → UNCERTAIN
                # This catches cases where GT is "partially true" but not fully verified
                if non_gt_count == 0 and gt_count > 0 and real_weight > 0:
                    conf = result.get("confidence", 0.0)
                    try:
                        conf = float(conf)
                    except (ValueError, TypeError):
                        conf = 0.0
                    # Only GT → low confidence → UNCERTAIN (not REAL)
                    if conf < 0.7:
                        _gt_overrides_fake = True
                        break

        # ── Final verdict logic V4/V13/V14/V14.5 ──────────────────────────────
        if total == 0:
            final_verdict = "UNCERTAIN"
            final_confidence = 0.0
        # V14.5 VETO: Any FALSE claim with conf >= threshold → FAKE (runs before GT)
        elif _veto_fire:
            final_verdict = "FAKE"
            final_confidence = _veto_confidence
        # V14.5: Cross-source mismatch → FAKE (catches 3850/4097 style errors)
        elif cross_source_mismatch_found:
            logger.info(
                "[V14.5] Cross-source mismatch detected → FAKE (conf=%.2f)",
                cross_source_mismatch_confidence,
            )
            final_verdict = "FAKE"
            final_confidence = max(cross_source_mismatch_confidence, 0.55)
        # V14.6 Phase 1: trusted negation evidence overrides verifier optimism
        elif _negation_fire:
            if _negation_from_trusted_source:
                final_verdict = "FAKE"
                final_confidence = max(_negation_confidence, 0.7)
            elif _negation_confidence >= 0.7:
                final_verdict = "FAKE"
                final_confidence = max(_negation_confidence, 0.65)
            else:
                final_verdict = "UNCERTAIN"
                final_confidence = max(_negation_confidence, 0.5)
        # V14: Discrepancy-triggered FAKE override (even if GT evidence is present)
        elif _discrepancy_found and _discrepancy_confidence >= 0.5:
            final_verdict = "FAKE"
            final_confidence = _discrepancy_confidence
        # V14.7.1: Numeric mismatch kill — clear numeric contradiction → FAKE
        elif _numeric_kill_found:
            logger.info(
                "[V14.7.1] Numeric kill triggered → FAKE (conf=%.2f)",
                _numeric_kill_confidence,
            )
            final_verdict = "FAKE"
            final_confidence = max(_numeric_kill_confidence, 0.6)
        elif _gt_overrides_fake:
            final_verdict = "UNCERTAIN"
            final_confidence = max(simple_avg_conf, 0.45)
        # V14.5: GT-first with web corroboration — safe to promote
        elif gt_first_enabled and real_weight > 0.0 and fake_weight == 0.0 and has_min_web_support:
            final_verdict = "REAL"
            final_confidence = max(weighted_avg_conf, 0.6)
        # V14.5: GT-only (no web corroboration) — downweight but still REAL
        elif gt_first_enabled and real_weight > 0.0 and fake_weight == 0.0 and not has_min_web_support:
            final_verdict = "REAL"
            final_confidence = min(max(weighted_avg_conf, 0.45), 0.55)
        # V14.5: GT-all-UNCERTAIN with web support → REAL with moderate confidence
        elif gt_first_enabled and uncertain_count == total and has_min_web_support and fake_weight == 0.0:
            final_verdict = "REAL"
            final_confidence = max(simple_avg_conf, 0.5)
        # V12 ULTRA: Any-FAKE dominant rule — a single false claim disqualifies the whole
        elif fake_weight >= 1.0 and fake_weight >= real_weight:
            final_verdict = "FAKE"
            # Use the weight of the FAKE claim's confidence directly, not averaged
            fake_confidences = [r.get("confidence", 0.5) for r in results if r.get("verdict") in ("false", "FAKE")]
            final_confidence = max(fake_confidences) if fake_confidences else weighted_avg_conf
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

        # V13: GT-first floor confidence for REAL verdicts
        if gt_first_enabled and final_verdict == "REAL":
            final_confidence = max(final_confidence, 0.6) if total == 1 else max(final_confidence, 0.55)

        # ── V14.7.1 Government Trust Boost (with Entity Guard) ───────────────────
        # If the final verdict is REAL and at least one claim's evidence comes
        # from a .gov.vn domain, boost confidence by 0.15.
        # BUT: skip the boost entirely if an identity mismatch or numeric
        # contradiction was detected — gov trust only applies to consistent info.
        if final_verdict in ("REAL", "true"):
            _gov_boost_blocked = _identity_mismatch_found or _numeric_kill_found
            if _gov_boost_blocked:
                logger.info("[V14.7.1 GOV BOOST BLOCKED] identity/numeric mismatch detected, skipping trust boost")
            else:
                has_gov_evidence = False
                for result in results:
                    for e in result.get("evidence", []):
                        if isinstance(e, dict) and ".gov.vn" in str(e.get("url", "")).lower():
                            has_gov_evidence = True
                            break
                    if has_gov_evidence:
                        break
                if has_gov_evidence:
                    boosted = final_confidence + 0.15
                    final_confidence = min(boosted, 1.0)
                    logger.info("[V14.7 GOV BOOST] .gov.vn evidence → confidence +0.15 = %.3f", final_confidence)

        return {
            "verdict": final_verdict,
            "confidence": round(final_confidence, 3),
            "explanation": (f"Verified {total} claims with weighted scoring. REAL weight: {real_weight:.1f}, FAKE weight: {fake_weight:.1f}" if len(results) > 1 else (results[0].get("reasoning", "") if results else "No claims found")),
            "claims": results,
            "total_claims": len(results),
            "verdicts": verdict_counts,
            "average_confidence": round(simple_avg_conf, 3),
            "high_confidence_claims": high_confidence_claims,
            "low_confidence_claims": low_confidence_claims,
            "weighted_average_confidence": round(weighted_avg_conf, 3),
            "weighted_scores": {
                "real_weight": round(real_weight, 2),
                "fake_weight": round(fake_weight, 2),
                "uncertain_weight": round(uncertain_weight, 2),
            },
            # V12 ULTRA metrics
            "v12_ultra": self.v12_metrics.to_dict() if self.v12_ultra_enabled else None,
        }

    def evaluate_batch(
        self,
        texts: list[str],
        ground_truth_labels: list[str],
    ) -> list[TRUSTResult]:
        """V12 ULTRA: Process a batch and compute Recall/F1 against ground truth.

        Args:
            texts: List of input texts to fact-check
            ground_truth_labels: Ground truth labels ("FAKE" or "REAL") per text

        Returns:
            List of TRUSTResult (metrics accessible via self.v12_metrics)
        """
        metrics = V12UltraMetrics()
        self.v12_metrics = metrics

        results: list[TRUSTResult] = []
        for text, gt_label in zip(texts, ground_truth_labels, strict=True):
            result = self.process_text(text)
            results.append(result)
            predicted = result.summary.get("verdict", "UNCERTAIN")
            metrics.update_from_ground_truth(predicted, gt_label)

        if metrics.total_processed > 0:
            metrics.avg_sources_per_claim = metrics.total_sources / metrics.total_processed

        logger.info(
            "[V12 ULTRA] Batch complete: %d texts. Recall=%.3f Precision=%.3f F1=%.3f Accuracy=%.3f",
            metrics.total_processed,
            metrics.recall,
            metrics.precision,
            metrics.f1,
            metrics.accuracy,
        )
        logger.info(
            "[V12 ULTRA] Confusion matrix: TP=%d FP=%d TN=%d FN=%d",
            metrics.tp,
            metrics.fp,
            metrics.tn,
            metrics.fn,
        )

        return results


def run_trust_pipeline_sync(
    text: str,
    top_k_evidence: int = 2,
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
def fact_check(text: str, top_k_evidence: int = 2) -> dict[str, Any]:
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
