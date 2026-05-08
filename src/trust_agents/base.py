from abc import ABC, abstractmethod

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    content: str
    source: str
    url: str | None = None
    authority: str = "medium"  # high, medium, low

class ClaimResult(BaseModel):
    claim: str
    verdict: str  # REAL, FAKE, UNCERTAIN, UNKNOWN
    confidence: float
    reasoning: str
    evidence: list[EvidenceItem] = []

class AgentLog(BaseModel):
    time: str
    level: str  # INFO, SUCCESS, ERROR, WARN
    agent: str
    msg: str

class AgentResponse(BaseModel):
    verdict: str
    confidence: float
    summary: str
    claims: list[ClaimResult]
    logs: list[AgentLog]
    processingMs: int

class BaseAgentBackend(ABC):
    """Interface chuẩn cho mọi backend agent."""

    @abstractmethod
    def analyze(self, text: str) -> AgentResponse:
        """Phân tích văn bản, trả về kết quả chuẩn hóa."""
        pass

    @abstractmethod
    def get_status(self) -> dict:
        """Trả về trạng thái agent (nodes, health, v.v.)."""
        pass
