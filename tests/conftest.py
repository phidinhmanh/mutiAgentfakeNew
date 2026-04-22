"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock
from unittest.mock import patch

import pytest


# === Sample Data Fixtures ===


@pytest.fixture
def sample_vietnamese_text() -> str:
    """Sample Vietnamese text for testing."""
    return """
    Theo báo cáo của Bộ Y tế, tính đến ngày 15 tháng 3 năm 2024,
    số ca mắc COVID-19 tại Việt Nam đã giảm 30% so với tháng trước.
    Ông Nguyễn Thanh Long, Bộ trưởng Bộ Y tế cho biết:
    "Chúng tôi đã kiểm soát được dịch bệnh."
    """


@pytest.fixture
def sample_claim() -> str:
    """Sample claim for testing."""
    return "Việt Nam là nước có tốc độ tăng trưởng kinh tế nhanh nhất ASEAN"


@pytest.fixture
def sample_evidence() -> list[dict[str, Any]]:
    """Sample evidence list for testing."""
    return [
        {
            "content": "Việt Nam đạt tăng trưởng 8% trong năm 2023, cao nhất ASEAN",
            "source": "vnexpress",
            "score": 0.9,
        },
        {
            "content": "Theo số liệu của World Bank, GDP Việt Nam tăng trưởng 5.6%",
            "source": "worldbank",
            "score": 0.7,
        },
    ]


@pytest.fixture
def sample_claims() -> list[dict[str, Any]]:
    """Sample claims with FACT/OPINION classification."""
    return [
        {
            "text": "Số ca mắc COVID-19 giảm 30% so với tháng trước",
            "type": "FACT",
            "verifiable": True,
        },
        {
            "text": "Tôi nghĩ chúng ta đã kiểm soát được dịch bệnh",
            "type": "OPINION",
            "verifiable": False,
        },
    ]


@pytest.fixture
def sample_verdicts() -> list[dict[str, Any]]:
    """Sample verdicts for aggregation testing."""
    return [
        {"verdict": "REAL", "confidence": 0.8, "reasoning": "Evidence supports claim"},
        {"verdict": "REAL", "confidence": 0.7, "reasoning": "Consistent with sources"},
        {"verdict": "FAKE", "confidence": 0.6, "reasoning": "Some inconsistency"},
    ]


# === NVIDIA NIM Client Mock Fixtures ===


@pytest.fixture
def mock_nvidia_client() -> Mock:
    """Mock NVIDIA NIM API client."""
    client = Mock()
    client.invoke = Mock(
        return_value='{"verdict": "REAL", "confidence": 0.8, "reasoning": "Test reasoning", "citations": []}'
    )
    client.stream = Mock(return_value=iter(['{"verdict": "REAL', '", "confidence": 0.8}']))
    return client


@pytest.fixture
def mock_nvidia_client_fake() -> Mock:
    """Mock NVIDIA NIM API client returning FAKE verdict."""
    client = Mock()
    client.invoke = Mock(
        return_value='{"verdict": "FAKE", "confidence": 0.85, "reasoning": "Evidence contradicts claim", "citations": []}'
    )
    client.stream = Mock(return_value=iter(['{"verdict": "FAKE', '", "confidence": 0.85}']))
    return client


@pytest.fixture
def mock_nvidia_client_unverifiable() -> Mock:
    """Mock NVIDIA NIM API client returning UNVERIFIABLE verdict."""
    client = Mock()
    client.invoke = Mock(
        return_value='{"verdict": "UNVERIFIABLE", "confidence": 0.0, "reasoning": "Not enough evidence", "citations": []}'
    )
    client.stream = Mock(return_value=iter(['{"verdict": "UNVERIFIABLE', '", "confidence": 0.0}']))
    return client


# === External API Mock Fixtures ===


@pytest.fixture
def mock_serper_api(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock Serper API responses."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={
        "organic": [
            {
                "title": "Vietnam Economy Grows",
                "snippet": "Vietnam GDP growth 8% in 2023",
                "link": "https://vnexpress.net/test",
            },
            {
                "title": "Economic Report",
                "snippet": "Vietnam fastest growing economy in ASEAN",
                "link": "https://tuoitre.vn/test",
            },
        ]
    })
    mock_response.raise_for_status = Mock()

    def mock_post(*args: Any, **kwargs: Any) -> Mock:
        return mock_response

    monkeypatch.setattr("requests.post", mock_post)
    return mock_response


@pytest.fixture
def mock_tavily_api(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock Tavily API responses."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={
        "results": [
            {
                "title": "Vietnam Economy",
                "content": "Vietnam GDP grew 8% in 2023, fastest in ASEAN",
                "url": "https://tavily.com/test",
                "score": 0.9,
            }
        ]
    })
    mock_response.raise_for_status = Mock()

    def mock_post(*args: Any, **kwargs: Any) -> Mock:
        return mock_response

    monkeypatch.setattr("requests.post", mock_post)
    return mock_response


# === Sentence Transformer Mock Fixtures ===


@pytest.fixture
def mock_sentence_transformer() -> Mock:
    """Mock SentenceTransformer for vector store tests."""
    with patch("sentence_transformers.SentenceTransformer") as mock:
        instance = Mock()
        instance.encode = Mock(return_value=[[0.1, 0.2, 0.3]])
        instance.get_sentence_embedding_dimension = Mock(return_value=384)
        mock.return_value = instance
        yield mock


# === Underthesea Mock Fixtures ===


@pytest.fixture
def mock_underthesea(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock underthesea functions."""
    mock_sent_tokenize = Mock(return_value=[
        "Theo báo cáo của Bộ Y tế.",
        "Tính đến ngày 15 tháng 3 năm 2024.",
        "Số ca mắc COVID-19 tại Việt Nam giảm 30% so với tháng trước.",
        "Tôi nghĩ chúng ta đã kiểm soát được dịch bệnh.",
    ])
    mock_word_tokenize = Mock(return_value=["Theo", "báo", "cáo"])
    mock_pos_tag = Mock(return_value=[
        ("Theo", "E"), ("báo", "N"), ("cáo", "N"),
    ])

    monkeypatch.setattr("underthesea.sent_tokenize", mock_sent_tokenize)
    monkeypatch.setattr("underthesea.word_tokenize", mock_word_tokenize)
    monkeypatch.setattr("underthesea.pos_tag", mock_pos_tag)

    return Mock(
        sent_tokenize=mock_sent_tokenize,
        word_tokenize=mock_word_tokenize,
        pos_tag=mock_pos_tag,
    )


# === HuggingFace Model Mock Fixtures ===


@pytest.fixture
def mock_transformers_tokenizer() -> Mock:
    """Mock HuggingFace tokenizer."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock:
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": [[1, 2, 3, 4, 5]]}
        tokenizer.pad = Mock(return_value={"input_ids": [[1, 2, 3, 4, 5]]})
        mock.return_value = tokenizer
        yield mock


@pytest.fixture
def mock_transformers_model() -> Mock:
    """Mock HuggingFace model."""
    with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock:
        model = Mock()
        model.eval = Mock()

        outputs = Mock()
        outputs.logits = Mock()
        outputs.logits.argmax = Mock(return_value=0)
        outputs.logits.softmax = Mock(return_value=[[0.7, 0.3]])
        model.return_value = outputs

        mock.return_value = model
        yield mock


# === Reusable Mock Response Classes ===


class MockSerperResponse:
    """Reusable Serper API response mock."""

    @staticmethod
    def get_success_response() -> dict[str, Any]:
        return {
            "organic": [
                {
                    "title": "Vietnam Economy Grows",
                    "snippet": "Vietnam GDP growth 8% in 2023",
                    "link": "https://vnexpress.net/test",
                }
            ]
        }

    @staticmethod
    def get_empty_response() -> dict[str, Any]:
        return {"organic": []}


class MockTavilyResponse:
    """Reusable Tavily API response mock."""

    @staticmethod
    def get_success_response() -> dict[str, Any]:
        return {
            "results": [
                {
                    "title": "Vietnam Economy",
                    "content": "Vietnam GDP grew 8% in 2023",
                    "url": "https://tavily.com/test",
                    "score": 0.9,
                }
            ]
        }

    @staticmethod
    def get_empty_response() -> dict[str, Any]:
        return {"results": []}


class MockNVIDIAResponse:
    """Reusable NVIDIA NIM API response mock."""

    @staticmethod
    def get_real_verdict() -> str:
        return (
            '{"verdict": "REAL", "confidence": 0.8, '
            '"reasoning": "Evidence supports claim", '
            '"citations": [{"evidence_id": 0, "quote_text": "Vietnam GDP grew 8%"}]}'
        )

    @staticmethod
    def get_fake_verdict() -> str:
        return (
            '{"verdict": "FAKE", "confidence": 0.85, '
            '"reasoning": "Evidence contradicts claim", '
            '"citations": []}'
        )

    @staticmethod
    def get_unverifiable_verdict() -> str:
        return (
            '{"verdict": "UNVERIFIABLE", "confidence": 0.0, '
            '"reasoning": "Not enough evidence", '
            '"citations": []}'
        )

    @staticmethod
    def get_invalid_json() -> str:
        return "This is not valid JSON response from LLM"


# === Orchestrator Fixtures for Integration/Acceptance Tests ===

class MockAgentResponses:
    """Mock responses for agent pipeline testing."""

    @staticmethod
    def claim_extractor_success() -> str:
        """Return valid claims JSON."""
        return '{"claims": ["Việt Nam đạt tăng trưởng 8% trong năm 2023", "GDP Việt Nam cao nhất ASEAN"]}'

    @staticmethod
    def claim_extractor_multiple() -> str:
        """Return multiple claims including opinions."""
        return '{"claims": ["Số ca mắc COVID-19 giảm 30%", "Chúng tôi đã kiểm soát được dịch", "Tốc độ tăng trưởng đạt 8%"]}'

    @staticmethod
    def claim_extractor_empty() -> str:
        """Return empty claims list."""
        return '{"claims": []}'

    @staticmethod
    def evidence_retriever_real() -> str:
        """Return evidence supporting real claim."""
        return json.dumps({
            "evidence": [
                {"content": "Việt Nam đạt tăng trưởng 8% trong quý 3/2023, cao nhất ASEAN", "source": "vnexpress", "score": 0.92},
                {"content": "Theo WB, GDP VN 2023 đạt 5.6%, thuộc nhóm cao nhất khu vực", "source": "worldbank", "score": 0.88},
            ]
        })

    @staticmethod
    def evidence_retriever_fake() -> str:
        """Return evidence contradicting claim."""
        return json.dumps({
            "evidence": [
                {"content": "Việt Nam tăng trưởng 4.2% trong năm 2023, không phải cao nhất", "source": "tuoitre", "score": 0.85},
                {"content": "Campuchia đạt 5.6%, cao hơn Việt Nam", "source": "vietnamnet", "score": 0.82},
            ]
        })

    @staticmethod
    def verifier_real() -> str:
        """Return REAL verdict."""
        return json.dumps({
            "verdict": "true",
            "confidence": 0.85,
            "reasoning": "Evidence strongly supports the claim. Multiple sources confirm Vietnam's 8% growth.",
            "label": "true"
        })

    @staticmethod
    def verifier_fake() -> str:
        """Return FAKE verdict."""
        return json.dumps({
            "verdict": "false",
            "confidence": 0.90,
            "reasoning": "Evidence contradicts the claim. Actual growth is 4.2%, not 8%.",
            "label": "false"
        })

    @staticmethod
    def verifier_unverifiable() -> str:
        """Return UNVERIFIABLE verdict."""
        return json.dumps({
            "verdict": "uncertain",
            "confidence": 0.3,
            "reasoning": "Not enough evidence to verify this claim.",
            "label": "uncertain"
        })

    @staticmethod
    def verifier_with_citations() -> str:
        """Return verdict with citations."""
        return json.dumps({
            "verdict": "true",
            "confidence": 0.85,
            "reasoning": "Evidence from multiple sources supports claim",
            "label": "true",
            "citations": [
                {"evidence_id": 0, "quote_text": "Vietnam GDP grew 8%"},
                {"evidence_id": 1, "quote_text": "Highest in ASEAN"}
            ]
        })

    @staticmethod
    def explainer_report() -> str:
        """Return explainer report."""
        return json.dumps({
            "summary": "Claim verified as REAL with 85% confidence",
            "explanation": "Multiple authoritative sources confirm Vietnam's economic growth",
            "evidence_count": 2
        })


@pytest.fixture
def mock_claim_extractor_agent(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock claim extractor agent for integration tests."""
    mock = Mock()
    mock.return_value = [
        "Việt Nam đạt tăng trưởng 8% trong năm 2023",
        "GDP Việt Nam cao nhất ASEAN"
    ]
    monkeypatch.setattr(
        "trust_agents.orchestrator.run_claim_extractor_agent_sync",
        mock
    )
    return mock


@pytest.fixture
def mock_claim_extractor_multiple(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock claim extractor returning multiple claims."""
    mock = Mock()
    mock.return_value = [
        "Số ca mắc COVID-19 giảm 30%",
        "Chúng tôi đã kiểm soát được dịch",
        "Tốc độ tăng trưởng đạt 8%"
    ]
    monkeypatch.setattr(
        "trust_agents.orchestrator.run_claim_extractor_agent_sync",
        mock
    )
    return mock


@pytest.fixture
def mock_claim_extractor_empty(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock claim extractor returning empty list."""
    mock = Mock()
    mock.return_value = []
    monkeypatch.setattr(
        "trust_agents.orchestrator.run_claim_extractor_agent_sync",
        mock
    )
    return mock


@pytest.fixture
def mock_evidence_retriever(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock evidence retriever agent for integration tests."""
    mock = Mock()
    mock.return_value = [
        {"content": "Việt Nam đạt tăng trưởng 8%", "source": "vnexpress", "score": 0.92},
        {"content": "GDP cao nhất ASEAN", "source": "worldbank", "score": 0.88}
    ]
    monkeypatch.setattr(
        "trust_agents.orchestrator.run_evidence_retrieval_agent_sync",
        mock
    )
    return mock


@pytest.fixture
def mock_evidence_retriever_empty(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock evidence retriever returning empty list."""
    mock = Mock()
    mock.return_value = []
    monkeypatch.setattr(
        "trust_agents.orchestrator.run_evidence_retrieval_agent_sync",
        mock
    )
    return mock


@pytest.fixture
def mock_verifier_agent(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock verifier agent for integration tests."""
    mock = Mock()
    mock.return_value = {
        "verdict": "true",
        "confidence": 0.85,
        "label": "true",
        "reasoning": "Evidence supports claim"
    }
    monkeypatch.setattr(
        "trust_agents.orchestrator.run_verifier_agent_sync",
        mock
    )
    return mock


@pytest.fixture
def mock_verifier_agent_fake(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock verifier agent returning FAKE verdict."""
    mock = Mock()
    mock.return_value = {
        "verdict": "false",
        "confidence": 0.90,
        "label": "false",
        "reasoning": "Evidence contradicts claim"
    }
    monkeypatch.setattr(
        "trust_agents.orchestrator.run_verifier_agent_sync",
        mock
    )
    return mock


@pytest.fixture
def mock_explainer_agent(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock explainer agent for integration tests."""
    mock = Mock()
    mock.return_value = {
        "summary": "Claim verified as REAL",
        "explanation": "Multiple sources confirm",
        "verdict": "true",
        "confidence": 0.85
    }
    monkeypatch.setattr(
        "trust_agents.orchestrator.run_explainer_agent_sync",
        mock
    )
    return mock


# === Golden Samples for Acceptance Tests ===

@pytest.fixture
def golden_samples_vietnamese() -> list[dict[str, Any]]:
    """Golden samples for acceptance testing with Vietnamese news."""
    return [
        {
            "id": "sample_real_1",
            "text": "Theo báo cáo của WB, Việt Nam đạt tăng trưởng GDP 8% trong quý 3 năm 2023, cao nhất ASEAN.",
            "expected_verdict": "true",
            "expected_claims": 1,
            "description": "Tin thật với số liệu cụ thể từ nguồn uy tín"
        },
        {
            "id": "sample_fake_1",
            "text": "Việt Nam đã phát hiện người ngoài hành tinh tại Hà Nội và chính phủ đang che giấu thông tin này.",
            "expected_verdict": "false",
            "expected_claims": 1,
            "description": "Tin giả với thông tin vô lý"
        },
        {
            "id": "sample_unverifiable_1",
            "text": "Một nguồn tin cho biết lãnh đạo đất nước sẽ có cuộc họp quan trọng vào tuần tới nhưng không tiết lộ chi tiết.",
            "expected_verdict": "uncertain",
            "expected_claims": 1,
            "description": "Tin không thể xác minh do thiếu chi tiết cụ thể"
        },
        {
            "id": "sample_real_2",
            "text": "Bộ Y tế công bố số ca mắc COVID-19 ngày 15/3/2024 giảm 30% so với tuần trước, xuống còn 1,500 ca.",
            "expected_verdict": "true",
            "expected_claims": 1,
            "description": "Tin thật với số liệu chính thức từ Bộ Y tế"
        },
        {
            "id": "sample_fake_2",
            "text": "Tỷ lệ thất nghiệp tại Việt Nam đạt 50% trong năm 2024, cao nhất thế giới.",
            "expected_verdict": "false",
            "expected_claims": 1,
            "description": "Tin giả với số liệu phi lý"
        }
    ]


@pytest.fixture
def sample_orchestrator_result() -> dict[str, Any]:
    """Sample orchestrator result structure for testing."""
    return {
        "original_text": "Việt Nam đạt tăng trưởng 8%",
        "claims": ["Việt Nam đạt tăng trưởng 8%"],
        "results": [
            {
                "claim": "Việt Nam đạt tăng trưởng 8%",
                "verdict": "true",
                "confidence": 0.85,
                "label": "true",
                "reasoning": "Evidence supports claim",
                "summary": "Claim verified",
                "explanation": "Multiple sources confirm"
            }
        ],
        "summary": {
            "total_claims": 1,
            "verdicts": {"true": 1, "false": 0, "uncertain": 0, "error": 0},
            "average_confidence": 0.85,
            "high_confidence_claims": 1,
            "low_confidence_claims": 0
        }
    }