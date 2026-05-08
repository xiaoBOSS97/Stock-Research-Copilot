from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import price_loader


class FakeTicker:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.info = {
            "longName": "Example Corp",
            "sector": "Technology",
            "industry": "Software",
            "marketCap": 123,
            "currency": "USD",
            "forwardEps": 5.0,
            "totalRevenue": 1_000.0,
            "sharesOutstanding": 10.0,
        }

    def history(self, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [2.0, 1.0],
                "High": [3.0, 2.0],
                "Low": [1.5, 0.5],
                "Close": [2.5, 1.5],
                "Adj Close": [2.5, 1.5],
                "Volume": [2000, 1000],
            },
            index=pd.to_datetime(["2025-01-02", "2025-01-01"]),
        )


class EmptyTicker(FakeTicker):
    def history(self, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        return pd.DataFrame()


class MissingColumnTicker(FakeTicker):
    def history(self, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        return pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5]},
            index=pd.date_range("2025-01-01", periods=1, freq="D"),
        )


def test_get_price_history_normalizes_ticker_and_columns(monkeypatch) -> None:
    monkeypatch.setattr(price_loader.yf, "Ticker", FakeTicker)

    result = price_loader.get_price_history(" aapl ", period="1d", interval="1d")

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert result.index.name == "Date"
    assert result.index.is_monotonic_increasing


def test_get_company_profile_maps_basic_fields(monkeypatch) -> None:
    monkeypatch.setattr(price_loader.yf, "Ticker", FakeTicker)

    profile = price_loader.get_company_profile("AAPL")

    assert profile["ticker"] == "AAPL"
    assert profile["name"] == "Example Corp"
    assert profile["currency"] == "USD"
    assert profile["forwardEps"] == pytest.approx(5)
    assert profile["totalRevenue"] == pytest.approx(1_000)
    assert profile["sharesOutstanding"] == pytest.approx(10)


def test_get_price_history_raises_for_empty_data(monkeypatch) -> None:
    monkeypatch.setattr(price_loader.yf, "Ticker", EmptyTicker)

    with pytest.raises(price_loader.PriceDataError, match="No price history returned"):
        price_loader.get_price_history("AAPL")


def test_get_price_history_raises_for_missing_columns(monkeypatch) -> None:
    monkeypatch.setattr(price_loader.yf, "Ticker", MissingColumnTicker)

    with pytest.raises(price_loader.PriceDataError, match="missing columns"):
        price_loader.get_price_history("AAPL")


def test_validate_ticker_returns_false_for_unavailable_ticker(monkeypatch) -> None:
    monkeypatch.setattr(price_loader.yf, "Ticker", EmptyTicker)

    assert price_loader.validate_ticker("NOPE") is False


def test_get_price_history_can_cache_normalized_csv(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(price_loader.yf, "Ticker", FakeTicker)

    result = price_loader.get_price_history("AAPL", period="5d", interval="1d", cache=True, cache_dir=tmp_path)

    expected_path = tmp_path / "AAPL_5d_1d_prices.csv"
    assert expected_path.exists()
    assert len(result) == 2


def test_clean_ticker_rejects_empty_ticker() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        price_loader.get_price_history(" ")
