"""Lightweight performance/regression tests for FastAPI endpoints."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────



@pytest.fixture
def fast_mock_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock all agent functions with minimal delay for performance testing."""
    mock_claim_extractor = Mock(return_value=["Claim 1"])
    mock_evidence_retriever = Mock(return_value=[{"content": "Ev 1", "source": "src", "score": 0.9}])
    mock_verifier = Mock(
        return_value={
            "verdict": "true",
            "confidence": 0.9,
            "reasoning": "Reason",
            "label": "true",
        }
    )
    mock_explainer = Mock(
        return_value={
            "verdict": "true",
            "confidence": 0.9,
            "reasoning": "Reason",
            "summary": "Summary",
            "explanation": "Explanation",
        }
    )

    mock_orchestrator_class = Mock()
    mock_orch_instance = Mock()
    mock_orch_instance._create_summary.return_value = {
        "verdict": "REAL",
        "confidence": 0.9,
        "explanation": "Summary",
    }
    mock_orchestrator_class.return_value = mock_orch_instance

    monkeypatch.setattr(
        "trust_agents.agents.claim_extractor.run_claim_extractor_agent_sync",
        mock_claim_extractor,
    )
    monkeypatch.setattr(
        "trust_agents.agents.evidence_retrieval.run_evidence_retrieval_agent_sync",
        mock_evidence_retriever,
    )
    monkeypatch.setattr(
        "trust_agents.agents.verifier.run_verifier_agent_sync",
        mock_verifier,
    )
    monkeypatch.setattr(
        "trust_agents.agents.explainer.run_explainer_agent_sync",
        mock_explainer,
    )
    monkeypatch.setattr(
        "trust_agents.orchestrator.TRUSTOrchestrator",
        mock_orchestrator_class,
    )


# ────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────


@pytest.mark.performance
def test_health_endpoint_latency(api_client: TestClient) -> None:
    """Test that /api/health responds quickly."""
    iterations = 20
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter()
        response = api_client.get("/api/health")
        end = time.perf_counter()

        assert response.status_code == 200
        latencies.append((end - start) * 1000)  # ms

    avg_latency = sum(latencies) / len(latencies)
    # Target < 50ms locally, conservative 100ms for CI
    assert avg_latency < 100, f"Health endpoint too slow: {avg_latency:.2f}ms avg"


@pytest.mark.performance
def test_status_endpoint_latency(api_client: TestClient) -> None:
    """Test that /api/status responds quickly."""
    iterations = 20
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter()
        response = api_client.get("/api/status")
        end = time.perf_counter()

        assert response.status_code == 200
        latencies.append((end - start) * 1000)  # ms

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 100, f"Status endpoint too slow: {avg_latency:.2f}ms avg"


@pytest.mark.performance
def test_analyze_stream_latency_to_first_event(api_client: TestClient, fast_mock_agents: None) -> None:
    """Test time-to-first-event for /api/analyze SSE stream."""
    start = time.perf_counter()
    first_event_time = 0

    with api_client.stream("POST", "/api/analyze", json={"text": "Test performance"}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line:
                first_event_time = (time.perf_counter() - start) * 1000  # ms
                break

    assert first_event_time > 0, "No event received"
    # Thread pool startup + queue dispatch should be fast. Target < 250ms.
    assert first_event_time < 250, f"Time to first event too slow: {first_event_time:.2f}ms"


@pytest.mark.performance
def test_analyze_sequential_throughput(api_client: TestClient, fast_mock_agents: None) -> None:
    """Test that the backend handles sequential requests without leaking or hanging."""
    iterations = 10
    total_time = 0.0

    for i in range(iterations):
        start = time.perf_counter()
        events_received = 0
        with api_client.stream("POST", "/api/analyze", json={"text": f"Sequential test {i}"}) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line:
                    events_received += 1

        end = time.perf_counter()
        total_time += end - start
        assert events_received > 0, f"Request {i} received no events"

    avg_request_time = (total_time / iterations) * 1000  # ms
    # Sequential processing of mocked pipeline should be < 500ms per request.
    assert avg_request_time < 500, f"Sequential throughput too slow: {avg_request_time:.2f}ms avg"
