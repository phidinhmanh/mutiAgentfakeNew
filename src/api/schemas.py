from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    text: str

class EvidenceItem(BaseModel):
    content: str
    source: str
    url: str | None = None
    authority: str = "medium"

class ClaimResult(BaseModel):
    claim: str
    verdict: str
    confidence: float
    reasoning: str
    evidence: list[EvidenceItem] = []

class AgentLog(BaseModel):
    time: str
    level: str
    agent: str
    msg: str

class AgentResponse(BaseModel):
    verdict: str
    confidence: float
    summary: str
    claims: list[ClaimResult]
    logs: list[AgentLog]
    processingMs: int

class StatusResponse(BaseModel):
    agents: dict
    is_busy: bool
    health: str
