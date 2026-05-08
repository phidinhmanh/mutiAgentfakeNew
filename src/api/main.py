"""
FastAPI app - exposes TRUST agents via HTTP API with True SSE streaming.

Architecture:
- analyze() runs in a background thread
- Each agent step emits a log dict into an asyncio.Queue
- SSE generator reads from queue and yields events to client
- Result is sent after all logs (or on error)
"""
import asyncio
import concurrent.futures
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.main")

# Thread pool for blocking TRUST pipeline
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Global busy state & shared queue for SSE
_is_busy = False
_backend = None


def _get_backend():
    global _backend
    if _backend is None:
        from trust_agents.http_adapter import TRUSTBackend
        _backend = TRUSTBackend(top_k_evidence=2)
    return _backend


def _format_time() -> str:
    now = datetime.now()
    return now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"


class AnalysisRequest(BaseModel):
    text: str


# ─────────────────────────────────────────────────────────────────
# SSE Event format: "data: {json}\n\n"
# ─────────────────────────────────────────────────────────────────

async def _stream_trust_analysis(text: str) -> AsyncGenerator[str, None]:
    """
    Stream TRUST analysis via SSE.
    - Logs are yielded as they're produced (real-time)
    - Final result is yielded after all logs
    """
    from shared_fact_checking.llm_utils import get_domain_authority
    from trust_agents.agents.claim_extractor import run_claim_extractor_agent_sync
    from trust_agents.agents.evidence_retrieval import run_evidence_retrieval_agent_sync
    from trust_agents.agents.explainer import run_explainer_agent_sync
    from trust_agents.agents.verifier import run_verifier_agent_sync
    from trust_agents.http_adapter import _normalize_verdict
    from trust_agents.orchestrator import TRUSTOrchestrator

    queue: asyncio.Queue[tuple[str, dict] | str] = asyncio.Queue()

    # Capture the running event loop from the main thread
    loop = asyncio.get_running_loop()

    def _emit(event_type: str, data: dict):
        """Called from sync thread to enqueue SSE event."""
        try:
            loop.call_soon_threadsafe(
                lambda: queue.put_nowait((event_type, data))
            )
        except RuntimeError:
            pass  # Event loop might be closed

    def _emit_log(level: str, agent: str, msg: str):
        _emit("log", {
            "time": _format_time(),
            "level": level,
            "agent": agent,
            "msg": msg,
        })

    def analysis_thread():
        """Runs in thread pool — emits logs in real-time as steps complete."""
        global _is_busy
        _is_busy = True
        try:
            # Step 1: Start
            _emit_log("INFO", "Orchestrator", "Bắt đầu quy trình kiểm chứng TRUST...")

            # Step 2: Extract claims
            _emit_log("INFO", "Claim Extractor", "Đang trích xuất các tuyên bố factual...")
            try:
                claims = run_claim_extractor_agent_sync(text)
            except Exception as e:
                _emit_log("ERROR", "Claim Extractor", f"Lỗi trích xuất: {str(e)}")
                claims = []

            num_claims = len(claims)
            _emit_log("SUCCESS", "Claim Extractor", f"Đã trích xuất {num_claims} tuyên bố.")

            if not claims:
                _emit_log("WARN", "Orchestrator", "Không tìm thấy tuyên bố nào trong văn bản.")
                return

            # Step 3: Process each claim
            results = []
            for i, claim in enumerate(claims, 1):
                _emit_log("INFO", "Evidence Retriever", f"Đang tìm kiếm bằng chứng cho tuyên bố {i}/{num_claims}...")

                try:
                    evidence = run_evidence_retrieval_agent_sync(claim, top_k=2)
                except Exception as e:
                    _emit_log("ERROR", "Evidence Retriever", f"Lỗi tìm kiếm: {str(e)}")
                    evidence = []

                _emit_log("INFO", "Verifier", f"Đang kiểm chứng tuyên bố {i}/{num_claims}...")

                try:
                    if evidence:
                        verdict_data = run_verifier_agent_sync(claim, evidence)
                    else:
                        verdict_data = {
                            "verdict": "uncertain",
                            "confidence": 0.1,
                            "reasoning": "Không có bằng chứng để kiểm chứng.",
                        }
                except Exception as e:
                    _emit_log("ERROR", "Verifier", f"Lỗi kiểm chứng: {str(e)}")
                    verdict_data = {"verdict": "uncertain", "confidence": 0.0, "reasoning": str(e)}

                # Normalize verdict
                v_label = _normalize_verdict(verdict_data.get("verdict", "uncertain"))
                v_conf = float(verdict_data.get("confidence", 0.0))
                v_reasoning = verdict_data.get("reasoning", "")

                _emit_log(
                    "ERROR" if v_label == "FAKE" else "SUCCESS",
                    "Verifier",
                    f"Tuyên bố {i}: {v_label} ({v_conf*100:.1f}%)"
                )

                _emit_log("INFO", "Explainer", "Đang tạo báo cáo giải thích...")

                try:
                    report = run_explainer_agent_sync(claim, verdict_data, evidence)
                except Exception as e:
                    _emit_log("ERROR", "Explainer", f"Lỗi tạo báo cáo: {str(e)}")
                    report = verdict_data.copy()

                results.append({
                    "claim": claim,
                    "verdict": report.get("verdict", verdict_data.get("verdict", "uncertain")),
                    "confidence": float(report.get("confidence", verdict_data.get("confidence", 0.0))),
                    "reasoning": report.get("reasoning", report.get("summary", v_reasoning)),
                    "evidence": evidence if isinstance(evidence, list) else [],
                })

            # Step 4: Create summary
            _emit_log("INFO", "Orchestrator", "Đang tổng hợp kết quả cuối cùng...")

            try:
                orch = TRUSTOrchestrator(top_k_evidence=2)
                summary = orch._create_summary(results)
            except Exception as e:
                _emit_log("ERROR", "Orchestrator", f"Lỗi tổng hợp: {str(e)}")
                summary = {"verdict": "UNCERTAIN", "confidence": 0.0, "explanation": str(e)}

            final_verdict = _normalize_verdict(summary.get("verdict", "uncertain"))
            final_confidence = float(summary.get("confidence", 0.0))

            _emit_log(
                "SUCCESS",
                "Orchestrator",
                f"Phân tích hoàn tất. Kết luận: {final_verdict} — Độ tin cậy {final_confidence*100:.0f}%"
            )

            # Send final result
            _emit("result", {
                "verdict": final_verdict,
                "confidence": final_confidence,
                "summary": summary.get("explanation", ""),
                "claims": [
                    {
                        "claim": r["claim"],
                        "verdict": _normalize_verdict(r.get("verdict", "uncertain")),
                        "confidence": r.get("confidence", 0.0),
                        "reasoning": r.get("reasoning", ""),
                        "evidence": [
                            {
                                "content": ev.get("content", ev.get("text", "")),
                                "source": ev.get("source", ev.get("url", "")),
                                "url": ev.get("url", ""),
                                "authority": get_domain_authority(ev.get("url") or ""),
                            }
                            for ev in (r.get("evidence") or [])
                        ],
                    }
                    for r in results
                ],
                "processingMs": 0,
            })

        except Exception as e:
            logger.error(f"[ANALYSIS] Unhandled error: {e}", exc_info=True)
            _emit_log("ERROR", "Orchestrator", f"Lỗi nghiêm trọng: {str(e)}")
            _emit("result", {
                "verdict": "UNKNOWN",
                "confidence": 0.0,
                "summary": str(e),
                "claims": [],
                "processingMs": 0,
            })
        finally:
            _is_busy = False
            _emit("done", {})  # Signal end of stream

    # Start analysis in thread pool without blocking queue consumption
    loop = asyncio.get_running_loop()
    analysis_task = loop.run_in_executor(_executor, analysis_thread)

    # Stream events as they come from the queue
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=60.0)
        except asyncio.TimeoutError:
            # Keep the connection alive during long-running analysis.
            if analysis_task.done():
                break
            yield ": keep-alive\n\n"
            continue

        event_type, data = item

        if event_type == "log":
            yield f"data: {json.dumps({'type': 'log', 'payload': data}, ensure_ascii=False)}\n\n"
        elif event_type == "result":
            yield f"data: {json.dumps({'type': 'result', 'payload': data}, ensure_ascii=False)}\n\n"
        elif event_type == "done":
            break

    await analysis_task


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[API] Starting up — TRUST backend ready")
    yield
    logger.info("[API] Shutting down")
    _executor.shutdown(wait=False)


app = FastAPI(
    title="TRUST Agents API",
    description="Unified interface for multi-agent fact-checking with SSE streaming",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest) -> StreamingResponse:
    """
    Analyze text via TRUST pipeline with TRUE SSE streaming.

    Each agent step emits a log event as soon as it completes.
    The final result is sent after all logs have been streamed.
    """
    return StreamingResponse(
        _stream_trust_analysis(request.text),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status")
async def get_status():
    """Return agent status including busy state."""
    return {
        "agents": {
            "orchestrator": {"id": "orchestrator", "label": "Orchestrator", "status": "active" if not _is_busy else "busy", "tasks": 0},
            "claim-extractor": {"id": "claim-extractor", "label": "Claim Extractor", "status": "idle", "tasks": 0},
            "evidence-retriever": {"id": "evidence-retriever", "label": "Evidence Retriever", "status": "idle", "tasks": 0},
            "verifier": {"id": "verifier", "label": "Verifier", "status": "idle", "tasks": 0},
            "explainer": {"id": "explainer", "label": "Explainer", "status": "idle", "tasks": 0},
        },
        "is_busy": _is_busy,
        "health": "ok",
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})
