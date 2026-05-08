from __future__ import annotations

import pytest

from src.data_loader.transcript_loader import (
    TranscriptDownloadError,
    alpha_vantage_payload_to_text,
    download_alpha_vantage_transcript,
    normalize_quarter,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class FakeSession:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.params: dict[str, str] | None = None

    def get(self, url: str, params: dict[str, str], timeout: int) -> FakeResponse:
        self.params = params
        return FakeResponse(self.payload)


def test_normalize_quarter_accepts_valid_quarter() -> None:
    assert normalize_quarter(" 2024 q1 ") == "2024Q1"


def test_normalize_quarter_rejects_invalid_quarter() -> None:
    with pytest.raises(ValueError, match="YYYYQ#"):
        normalize_quarter("2024Q5")


def test_alpha_vantage_payload_to_text_handles_segment_list() -> None:
    text = alpha_vantage_payload_to_text(
        {
            "transcript": [
                {"speaker": "CEO", "content": "Demand was strong."},
                {"speaker": "CFO", "content": "Margins improved."},
            ]
        }
    )

    assert "CEO: Demand was strong." in text
    assert "CFO: Margins improved." in text


def test_download_alpha_vantage_transcript_uses_expected_params() -> None:
    session = FakeSession({"transcript": [{"speaker": "CEO", "content": "Revenue grew."}]})

    text = download_alpha_vantage_transcript(
        "aapl",
        "2024Q1",
        api_key="test-key",
        session=session,
    )

    assert "Revenue grew" in text
    assert session.params == {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": "AAPL",
        "quarter": "2024Q1",
        "apikey": "test-key",
    }


def test_download_alpha_vantage_transcript_raises_for_provider_message() -> None:
    session = FakeSession({"Information": "API limit reached"})

    with pytest.raises(TranscriptDownloadError, match="API limit"):
        download_alpha_vantage_transcript(
            "AAPL",
            "2024Q1",
            api_key="test-key",
            session=session,
        )
