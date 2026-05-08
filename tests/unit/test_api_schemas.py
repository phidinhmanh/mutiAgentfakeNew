from api.schemas import (
    AgentLog,
    AgentResponse,
    AnalysisRequest,
    ClaimResult,
    EvidenceItem,
    StatusResponse,
)


def test_analysis_request_valid():
    """Test AnalysisRequest accepts valid text."""
    req = AnalysisRequest(text="Việt Nam đạt tăng trưởng GDP 8% trong năm 2023.")
    assert req.text == "Việt Nam đạt tăng trưởng GDP 8% trong năm 2023."


def test_evidence_item_defaults():
    """Test EvidenceItem has expected defaults."""
    item = EvidenceItem(content="Báo cáo kinh tế", source="vnexpress.net")
    assert item.content == "Báo cáo kinh tế"
    assert item.source == "vnexpress.net"
    assert item.url is None
    assert item.authority == "medium"


def test_claim_result_valid():
    """Test ClaimResult validation."""
    result = ClaimResult(
        claim="GDP tăng trưởng 8%",
        verdict="REAL",
        confidence=0.9,
        reasoning="Nhiều nguồn uy tín xác nhận",
        evidence=[EvidenceItem(content="GDP là 8%", source="vnexpress.net", authority="high")],
    )
    assert result.claim == "GDP tăng trưởng 8%"
    assert result.verdict == "REAL"
    assert result.confidence == 0.9
    assert len(result.evidence) == 1
    assert result.evidence[0].authority == "high"


def test_agent_log_valid():
    """Test AgentLog schema."""
    log = AgentLog(time="12:00:00.000", level="INFO", agent="Orchestrator", msg="Started")
    assert log.time == "12:00:00.000"
    assert log.level == "INFO"
    assert log.agent == "Orchestrator"


def test_agent_response_valid():
    """Test AgentResponse schema with full data structure."""
    response = AgentResponse(
        verdict="REAL",
        confidence=0.95,
        summary="Thông tin hoàn toàn chính xác",
        claims=[],
        logs=[],
        processingMs=1200,
    )
    assert response.verdict == "REAL"
    assert response.confidence == 0.95
    assert response.processingMs == 1200


def test_status_response_valid():
    """Test StatusResponse schema."""
    status = StatusResponse(agents={"orchestrator": {"status": "idle"}}, is_busy=False, health="ok")
    assert status.is_busy is False
    assert status.health == "ok"
    assert "orchestrator" in status.agents
