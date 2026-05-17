from __future__ import annotations

import pytest

from src.data_loader.news_loader import (
    NewsDataError,
    alpha_vantage_news_payload_to_frame,
    fetch_alpha_vantage_news,
    load_news_cache,
    save_news_cache,
    summarize_news,
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


def sample_payload() -> dict[str, object]:
    return {
        "feed": [
            {
                "title": "Apple launches new device",
                "source": "Example News",
                "url": "https://example.com/aapl",
                "time_published": "20260507T143000",
                "summary": "Apple announced a new product.",
                "overall_sentiment_label": "Bullish",
                "overall_sentiment_score": "0.35",
                "ticker_sentiment": [{"ticker": "AAPL"}],
            },
            {
                "title": "Unrelated story",
                "source": "Other News",
                "url": "https://example.com/other",
                "time_published": "20260506T143000",
                "summary": "Another company announced something.",
                "overall_sentiment_label": "Neutral",
                "overall_sentiment_score": "0.00",
                "ticker_sentiment": [{"ticker": "MSFT"}],
            },
        ]
    }


def test_alpha_vantage_news_payload_to_frame_filters_and_normalizes() -> None:
    frame = alpha_vantage_news_payload_to_frame(sample_payload(), ticker="AAPL")

    assert len(frame) == 1
    assert frame.loc[0, "title"] == "Apple launches new device"
    assert frame.loc[0, "source"] == "Example News"
    assert frame.loc[0, "overall_sentiment_score"] == 0.35
    assert frame.loc[0, "published_at"].year == 2026


def test_fetch_alpha_vantage_news_uses_expected_params() -> None:
    session = FakeSession(sample_payload())

    frame = fetch_alpha_vantage_news(
        "aapl",
        limit=5,
        days_back=10,
        api_key="test-key",
        session=session,
    )

    assert len(frame) == 1
    assert session.params is not None
    assert session.params["function"] == "NEWS_SENTIMENT"
    assert session.params["tickers"] == "AAPL"
    assert session.params["limit"] == "5"
    assert session.params["apikey"] == "test-key"
    assert "time_from" in session.params


def test_fetch_alpha_vantage_news_raises_for_provider_message() -> None:
    session = FakeSession({"Information": "API limit reached"})

    with pytest.raises(NewsDataError, match="API limit"):
        fetch_alpha_vantage_news("AAPL", api_key="test-key", session=session)


def test_news_cache_roundtrip(tmp_path) -> None:
    frame = alpha_vantage_news_payload_to_frame(sample_payload(), ticker="AAPL")

    output_path = save_news_cache("aapl", frame, output_dir=tmp_path)
    loaded = load_news_cache("AAPL", input_dir=tmp_path)

    assert output_path == tmp_path / "AAPL_latest_news.json"
    assert loaded.loc[0, "title"] == "Apple launches new device"
    assert loaded.loc[0, "overall_sentiment_score"] == 0.35


def test_summarize_news_returns_report_ready_bullets() -> None:
    frame = alpha_vantage_news_payload_to_frame(sample_payload(), ticker="AAPL")

    summary = summarize_news(frame)

    assert "2026-05-07" in summary
    assert "Example News" in summary
    assert "Bullish" in summary
    assert "Apple launches new device" in summary
