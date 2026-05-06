from __future__ import annotations

import pandas as pd

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
        }

    def history(self, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [1.0],
                "High": [2.0],
                "Low": [0.5],
                "Close": [1.5],
                "Adj Close": [1.5],
                "Volume": [1000],
            },
            index=pd.date_range("2025-01-01", periods=1, freq="D"),
        )


def test_get_price_history_normalizes_ticker_and_columns(monkeypatch) -> None:
    monkeypatch.setattr(price_loader.yf, "Ticker", FakeTicker)

    result = price_loader.get_price_history(" aapl ", period="1d", interval="1d")

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert result.index.name == "Date"


def test_get_company_profile_maps_basic_fields(monkeypatch) -> None:
    monkeypatch.setattr(price_loader.yf, "Ticker", FakeTicker)

    profile = price_loader.get_company_profile("AAPL")

    assert profile.ticker == "AAPL"
    assert profile.name == "Example Corp"
    assert profile.currency == "USD"
