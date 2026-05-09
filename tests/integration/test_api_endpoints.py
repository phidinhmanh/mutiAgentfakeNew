"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_claim_extractor() -> Mock:
    """Mock claim extractor returning sample claims."""
    mock = Mock()
    mock.return_value = [
        "Việt Nam đạt tăng trưởng GDP 8% trong năm 2023",
        "Việt Nam là nền kinh tế tăng trưởng nhanh nhất ASEAN",
    ]
    return mock


@pytest.fixture
def mock_evidence_retriever() -> Mock:
    """Mock evidence retriever returning sample evidence."""
    mock = Mock()
    mock.return_value = [
        {
            "content": "Việt Nam đạt tăng trưởng 8% trong quý 3 năm 2023",
            "source": "vnexpress.net",
            "url": "https://vnexpress.net/test",
            "score": 0.92,
        },
        {
            "content": "Việt Nam tăng trưởng nhanh nhất ASEAN năm 2023",
            "source": "worldbank.org",
            "url": "https://worldbank.org/test",
            "score": 0.88,
        },
    ]
    return mock


@pytest.fixture
def mock_verifier_agent() -> Mock:
    """Mock verifier returning REAL verdict."""
    mock = Mock()
    mock.return_value = {
        "verdict": "true",
        "confidence": 0.85,
        "reasoning": "Evidence strongly supports the claim.",
        "label": "true",
    }
    return mock


@pytest.fixture
def mock_explainer_agent() -> Mock:
    """Mock explainer returning report."""
    mock = Mock()
    mock.return_value = {
        "verdict": "true",
        "confidence": 0.85,
        "reasoning": "Multiple sources confirm.",
        "summary": "Claim verified as TRUE with 85% confidence.",
        "explanation": "Multiple authoritative sources confirm Vietnam's economic growth.",
    }
    return mock


@pytest.fixture
def mock_orchestrator() -> Mock:
    """Mock TRUSTOrchestrator for summary creation."""
    mock_class = Mock()
    mock_instance = Mock()
    mock_instance._create_summary.return_value = {
        "verdict": "REAL",
        "confidence": 0.85,
        "explanation": "All claims verified as TRUE with high confidence.",
    }
    mock_class.return_value = mock_instance
    return mock_class


@pytest.fixture(autouse=True)
def mock_agents(
    monkeypatch: pytest.MonkeyPatch,
    mock_claim_extractor: Mock,
    mock_evidence_retriever: Mock,
    mock_verifier_agent: Mock,
    mock_explainer_agent: Mock,
    mock_orchestrator: Mock,
) -> None:
    """Patch all agent functions before each test."""
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
        mock_verifier_agent,
    )
    monkeypatch.setattr(
        "trust_agents.agents.explainer.run_explainer_agent_sync",
        mock_explainer_agent,
    )
    monkeypatch.setattr(
        "trust_agents.orchestrator.TRUSTOrchestrator",
        mock_orchestrator,
    )


# ────────────────────────────────────────────────────────────────
# Tests: /api/health
# ────────────────────────────────────────────────────────────────


def test_health_returns_ok(api_client: TestClient) -> None:
    """GET /api/health should return 200 with status ok."""
    response = api_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_no_auth_required(api_client: TestClient) -> None:
    """GET /api/health should be accessible without authentication."""
    response = api_client.get("/api/health")
    assert response.status_code == 200


# ────────────────────────────────────────────────────────────────
# Tests: /api/status
# ────────────────────────────────────────────────────────────────


def test_status_returns_200(api_client: TestClient) -> None:
    """GET /api/status should return 200."""
    response = api_client.get("/api/status")
    assert response.status_code == 200


def test_status_contains_agent_states(api_client: TestClient) -> None:
    """GET /api/status should include agent states."""
    data = api_client.get("/api/status").json()
    assert "agents" in data
    assert "is_busy" in data
    assert "health" in data


def test_status_agents_list(api_client: TestClient) -> None:
    """GET /api/status should list all pipeline agents."""
    data = api_client.get("/api/status").json()
    agents = data.get("agents", {})
    expected_agents = [
        "orchestrator",
        "claim-extractor",
        "evidence-retriever",
        "verifier",
        "explainer",
    ]
    for agent_id in expected_agents:
        assert agent_id in agents, f"Agent '{agent_id}' not found in status response"


def test_status_is_busy_field(api_client: TestClient) -> None:
    """GET /api/status should have is_busy as bool."""
    data = api_client.get("/api/status").json()
    assert isinstance(data["is_busy"], bool)


# ────────────────────────────────────────────────────────────────
# Tests: POST /api/analyze — validation
# ────────────────────────────────────────────────────────────────


def test_analyze_missing_body(api_client: TestClient) -> None:
    """POST /api/analyze with empty body should return 422."""
    response = api_client.post("/api/analyze")
    assert response.status_code == 422


def test_analyze_empty_text(api_client: TestClient) -> None:
    """POST /api/analyze with empty text should return 200 and handle it gracefully."""
    response = api_client.post("/api/analyze", json={"text": ""})
    # Since Pydantic doesn't strictly forbid empty text, it goes through and handles it.
    assert response.status_code == 200


def test_analyze_missing_text_field(api_client: TestClient) -> None:
    """POST /api/analyze with missing text field should return 422."""
    response = api_client.post("/api/analyze", json={})
    assert response.status_code == 422


# ────────────────────────────────────────────────────────────────
# Tests: POST /api/analyze — SSE streaming
# ────────────────────────────────────────────────────────────────


def test_analyze_returns_streaming_response(api_client: TestClient) -> None:
    """POST /api/analyze should return text/event-stream content type."""
    with api_client.stream(
        "POST",
        "/api/analyze",
        json={"text": "Việt Nam đạt tăng trưởng GDP 8%."},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


def test_analyze_sse_contains_log_events(api_client: TestClient) -> None:
    """POST /api/analyze should emit log events via SSE."""
    with api_client.stream(
        "POST",
        "/api/analyze",
        json={"text": "Việt Nam đạt tăng trưởng GDP 8%."},
    ) as response:
        lines = [line for line in response.iter_lines() if line]
        assert len(lines) > 0, "Expected SSE events but got empty stream"

        # At least one line should be a SSE data frame
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) > 0, f"Expected 'data:' SSE frames, got: {lines}"

        # Verify at least one log event
        has_log_event = any((json.loads(line.removeprefix("data: ")).get("type") == "log") for line in data_lines if _is_json(line.removeprefix("data: ")))
        assert has_log_event, f"Expected at least one log event, got data lines: {data_lines}"


def test_analyze_sse_contains_result_event(api_client: TestClient) -> None:
    """POST /api/analyze should emit a result event with verdict."""
    with api_client.stream(
        "POST",
        "/api/analyze",
        json={"text": "Việt Nam đạt tăng trưởng GDP 8%."},
    ) as response:
        data_lines = [json.loads(line.removeprefix("data: ")) for line in response.iter_lines() if line.startswith("data: ") and _is_json(line.removeprefix("data: "))]

        result_events = [e for e in data_lines if e.get("type") == "result"]
        assert len(result_events) == 1, f"Expected 1 result event, got: {data_lines}"

        payload = result_events[0].get("payload", {})
        assert "verdict" in payload
        assert "confidence" in payload


def test_analyze_unicode_vietnamese(api_client: TestClient) -> None:
    """POST /api/analyze should handle Vietnamese Unicode correctly."""
    with api_client.stream(
        "POST",
        "/api/analyze",
        json={"text": "Theo báo cáo của Bộ Y tế, Việt Nam kiểm soát dịch COVID-19 hiệu quả."},
    ) as response:
        assert response.status_code == 200
        data_lines = [json.loads(line.removeprefix("data: ")) for line in response.iter_lines() if line.startswith("data: ") and _is_json(line.removeprefix("data: "))]
        result_events = [e for e in data_lines if e.get("type") == "result"]
        assert len(result_events) == 1


def test_analyze_long_text(api_client: TestClient) -> None:
    """POST /api/analyze should handle moderately long input."""
    long_text = "Việt Nam đạt tăng trưởng GDP 8%. " * 100  # ~3.5k chars
    with api_client.stream(
        "POST",
        "/api/analyze",
        json={"text": long_text},
    ) as response:
        assert response.status_code == 200
        data_lines = [json.loads(line.removeprefix("data: ")) for line in response.iter_lines() if line.startswith("data: ") and _is_json(line.removeprefix("data: "))]
        result_events = [e for e in data_lines if e.get("type") == "result"]
        assert len(result_events) == 1


def test_analyze_multiple_claims(api_client: TestClient) -> None:
    """POST /api/analyze should process multiple claims in sequence."""
    with api_client.stream(
        "POST",
        "/api/analyze",
        json={"text": "Việt Nam đạt tăng trưởng 8%. Campuchia tăng 5%. Thái Lan tăng 3%."},
    ) as response:
        data_lines = [json.loads(line.removeprefix("data: ")) for line in response.iter_lines() if line.startswith("data: ") and _is_json(line.removeprefix("data: "))]
        result_events = [e for e in data_lines if e.get("type") == "result"]
        assert len(result_events) == 1

        payload = result_events[0].get("payload", {})
        claims = payload.get("claims", [])
        assert len(claims) == 2, f"Expected 2 claims, got {len(claims)}: {claims}"


# ────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────


def _is_json(s: str) -> bool:
    """Check if string is valid JSON."""
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False
