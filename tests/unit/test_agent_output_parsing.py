from __future__ import annotations

from trust_agents.parsing import (
    extract_json_object,
    parse_claims_payload,
    parse_evidence_payload,
)


class TestExtractJsonObject:
    """Test shared parsing helpers for TRUST agents."""

    def test_extracts_json_from_surrounding_text(self) -> None:
        payload = 'Some intro {"claims": ["a", "b"]} trailing text'

        result = extract_json_object(payload)

        assert result == {"claims": ["a", "b"]}

    def test_returns_none_when_no_json_found(self) -> None:
        assert extract_json_object("not json") is None


class TestParseClaimsPayload:
    def test_returns_cleaned_claims(self) -> None:
        payload = '{"claims": [" claim A ", "", null, "claim B"]}'

        result = parse_claims_payload(payload)

        assert result == ["claim A", "claim B"]

    def test_returns_empty_list_when_invalid(self) -> None:
        assert parse_claims_payload("oops") == []


class TestParseEvidencePayload:
    def test_returns_evidence_list(self) -> None:
        payload = '{"evidence": [{"text": "item 1"}, {"text": "item 2"}]}'

        result = parse_evidence_payload(payload)

        assert result == [{"text": "item 1"}, {"text": "item 2"}]

    def test_returns_empty_list_when_invalid(self) -> None:
        assert parse_evidence_payload("oops") == []
