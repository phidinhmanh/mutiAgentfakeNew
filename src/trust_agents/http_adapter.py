"""
TRUST Backend Adapter - implements BaseAgentBackend for TRUSTOrchestrator.
"""
import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from shared_fact_checking.llm_utils import get_domain_authority

from .base import AgentLog, AgentResponse, BaseAgentBackend, ClaimResult, EvidenceItem
from .orchestrator import TRUSTOrchestrator

logger = logging.getLogger("TRUST_agents.http_adapter")

# Node metadata - maps to UI NODE_INFO
NODE_METADATA = {
    "orchestrator": {"id": "orchestrator", "label": "Orchestrator", "status": "active", "tasks": 0},
    "claim-extractor": {"id": "claim-extractor", "label": "Claim Extractor", "status": "idle", "tasks": 0},
    "evidence-retriever": {"id": "evidence-retriever", "label": "Evidence Retriever", "status": "idle", "tasks": 0},
    "verifier": {"id": "verifier", "label": "Verifier", "status": "idle", "tasks": 0},
    "explainer": {"id": "explainer", "label": "Explainer", "status": "idle", "tasks": 0},
}


def _normalize_verdict(verdict: str) -> str:
    """Map TRUST verdict to standardized labels."""
    mapping = {
        "true": "REAL",
        "supported": "REAL",
        "real": "REAL",
        "false": "FAKE",
        "contradicted": "FAKE",
        "fake": "FAKE",
        "uncertain": "UNCERTAIN",
        "insufficient": "UNCERTAIN",
    }
    return mapping.get(verdict.lower(), "UNKNOWN")


def _extract_evidence(evidence_list: list[dict[str, Any]]) -> list[EvidenceItem]:
    """Convert orchestrator evidence format to EvidenceItem."""
    items = []
    for ev in evidence_list:
        if not isinstance(ev, dict):
            continue
        # Determine authority based on source URL
        url = ev.get("url", "")
        source = ev.get("source", "")
        authority = get_domain_authority(url) if url else "medium"

        items.append(EvidenceItem(
            content=ev.get("content", ev.get("text", "")),
            source=source or url or "Unknown",
            url=url,
            authority=authority,
        ))
    return items


def _format_time() -> str:
    """Format current time as HH:MM:SS.mmm"""
    now = datetime.now()
    return now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"


class TRUSTBackend(BaseAgentBackend):
    """TRUST orchestrator implementation of BaseAgentBackend."""

    def __init__(self, top_k_evidence: int = 2, log_callback: Callable[[dict], None] | None = None):
        self.top_k_evidence = top_k_evidence
        self.log_callback = log_callback
        self._orchestrator = None  # Lazy init
        logger.info("[TRUSTBackend] Initialized")

    @property
    def orchestrator(self) -> TRUSTOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = TRUSTOrchestrator(
                top_k_evidence=self.top_k_evidence,
                log_callback=self._emit_log,
            )
        return self._orchestrator

    def _emit_log(self, log_dict: dict):
        """Callback to store logs during processing."""
        if self.log_callback:
            self.log_callback(log_dict)

    def analyze(self, text: str) -> AgentResponse:
        """Run complete TRUST pipeline and normalize result."""
        start_time = time.time()
        logs: list[AgentLog] = []
        collected_logs: list[dict] = []

        # Use callback to collect logs
        self.log_callback = lambda log: collected_logs.append(log)

        # Re-create orchestrator with callback
        self._orchestrator = TRUSTOrchestrator(
            top_k_evidence=self.top_k_evidence,
            log_callback=self._emit_log,
        )

        # Run pipeline
        result = self.orchestrator.process_text(text)

        # Convert collected logs to AgentLog objects
        for log_dict in collected_logs:
            logs.append(AgentLog(**log_dict))

        # Process claims
        claims: list[ClaimResult] = []
        for i, claim_result in enumerate(result.results, 1):
            claim = claim_result.get("claim", "")
            verdict = _normalize_verdict(claim_result.get("verdict", "uncertain"))
            confidence = float(claim_result.get("confidence", 0.0))
            reasoning = claim_result.get("reasoning", claim_result.get("summary", ""))

            evidence_list = claim_result.get("evidence", [])
            evidence = _extract_evidence(evidence_list)

            claims.append(ClaimResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
                evidence=evidence,
            ))

        # Get summary
        summary = result.summary
        final_verdict = _normalize_verdict(summary.get("verdict", "uncertain"))
        final_confidence = float(summary.get("confidence", 0.0))
        explanation = summary.get("explanation", "")

        # Log: Final result if not already done
        if not any(l.msg.startswith("Phân tích hoàn tất") for l in logs):
            logs.append(AgentLog(
                time=_format_time(),
                level="SUCCESS",
                agent="Orchestrator",
                msg=f"Phân tích hoàn tất. Kết luận: {final_verdict} — Độ tin cậy {final_confidence*100:.0f}%"
            ))

        processing_ms = int((time.time() - start_time) * 1000)

        return AgentResponse(
            verdict=final_verdict,
            confidence=final_confidence,
            summary=explanation,
            claims=claims,
            logs=logs,
            processingMs=processing_ms,
        )

    def get_status(self) -> dict:
        """Return agent status metadata."""
        return {
            "agents": NODE_METADATA,
            "health": "ok",
        }

    async def analyze_stream(self, text: str, on_log: Callable[[dict], None]) -> AgentResponse:
        """Async version that streams logs via callback."""
        start_time = time.time()
        logs: list[AgentLog] = []
        collected_logs: list[dict] = []

        # Callback to stream logs back immediately
        def log_collector(log_dict: dict):
            collected_logs.append(log_dict)
            on_log(log_dict)  # Stream to client immediately

        # Create orchestrator with streaming callback
        orchestrator = TRUSTOrchestrator(
            top_k_evidence=self.top_k_evidence,
            log_callback=log_collector,
        )

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, orchestrator.process_text, text)

        # Convert collected logs
        for log_dict in collected_logs:
            logs.append(AgentLog(**log_dict))

        # Process claims
        claims: list[ClaimResult] = []
        for claim_result in result.results:
            verdict = _normalize_verdict(claim_result.get("verdict", "uncertain"))
            confidence = float(claim_result.get("confidence", 0.0))
            reasoning = claim_result.get("reasoning", claim_result.get("summary", ""))
            evidence_list = claim_result.get("evidence", [])

            claims.append(ClaimResult(
                claim=claim_result.get("claim", ""),
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
                evidence=_extract_evidence(evidence_list),
            ))

        # Get summary
        summary = result.summary
        final_verdict = _normalize_verdict(summary.get("verdict", "uncertain"))
        final_confidence = float(summary.get("confidence", 0.0))
        processing_ms = int((time.time() - start_time) * 1000)

        return AgentResponse(
            verdict=final_verdict,
            confidence=final_confidence,
            summary=summary.get("explanation", ""),
            claims=claims,
            logs=logs,
            processingMs=processing_ms,
        )
