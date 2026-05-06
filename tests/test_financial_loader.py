from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import financial_loader


class FakeTicker:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.income_stmt = pd.DataFrame(
            {
                "2025-12-31": [120.0, 30.0],
                "2024-12-31": [100.0, 20.0],
            },
            index=["Total Revenue", "Net Income"],
        )
        self.balance_sheet = pd.DataFrame(
            {
                "2025-12-31": [200.0],
                "2024-12-31": [180.0],
            },
            index=["Total Assets"],
        )
        self.cashflow = pd.DataFrame(
            {
                "2025-12-31": [25.0],
                "2024-12-31": [18.0],
            },
            index=["Free Cash Flow"],
        )


class EmptyTicker(FakeTicker):
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.income_stmt = pd.DataFrame()
        self.balance_sheet = pd.DataFrame()
        self.cashflow = pd.DataFrame()


def test_normalize_financials_returns_expected_statement_keys() -> None:
    raw = {
        "income_statement": pd.DataFrame(
            {"2024-12-31": ["100"], "2025-12-31": ["120"]},
            index=["Total Revenue"],
        )
    }

    normalized = financial_loader.normalize_financials(raw)

    assert set(normalized) == {"income_statement", "balance_sheet", "cash_flow"}
    assert normalized["income_statement"].columns[0] == pd.Timestamp("2025-12-31")
    assert normalized["income_statement"].loc["Total Revenue"].iloc[0] == pytest.approx(120)
    assert normalized["balance_sheet"].empty


def test_get_financial_statements_loads_and_normalizes(monkeypatch) -> None:
    monkeypatch.setattr(financial_loader.yf, "Ticker", FakeTicker)

    statements = financial_loader.get_financial_statements(" aapl ")

    assert statements["income_statement"].loc["Total Revenue"].iloc[0] == pytest.approx(120)
    assert statements["cash_flow"].loc["Free Cash Flow"].iloc[0] == pytest.approx(25)


def test_get_financial_statements_raises_for_empty_data(monkeypatch) -> None:
    monkeypatch.setattr(financial_loader.yf, "Ticker", EmptyTicker)

    with pytest.raises(financial_loader.FinancialDataError, match="No financial statements"):
        financial_loader.get_financial_statements("AAPL", use_cache_fallback=False)


def test_get_ttm_metrics_maps_latest_values(monkeypatch) -> None:
    monkeypatch.setattr(financial_loader.yf, "Ticker", FakeTicker)

    metrics = financial_loader.get_ttm_metrics("AAPL")

    assert metrics["revenue"] == pytest.approx(120)
    assert metrics["net_income"] == pytest.approx(30)
    assert metrics["free_cash_flow"] == pytest.approx(25)


def test_financial_statement_cache_roundtrip(tmp_path) -> None:
    raw = {
        "income_statement": pd.DataFrame(
            {"2025-12-31": [120.0]},
            index=["Total Revenue"],
        )
    }
    statements = financial_loader.normalize_financials(raw)

    paths = financial_loader.save_financial_statements(statements, "AAPL", output_dir=tmp_path)
    cached = financial_loader.load_cached_financials("AAPL", cache_dir=tmp_path)

    assert "income_statement" in paths
    assert cached["income_statement"].loc["Total Revenue"].iloc[0] == pytest.approx(120)


def test_get_financial_statements_uses_cache_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(financial_loader.yf, "Ticker", EmptyTicker)
    statements = financial_loader.normalize_financials(
        {
            "income_statement": pd.DataFrame(
                {"2025-12-31": [120.0]},
                index=["Total Revenue"],
            )
        }
    )
    financial_loader.save_financial_statements(statements, "AAPL", output_dir=tmp_path)

    result = financial_loader.get_financial_statements("AAPL", cache_dir=tmp_path)

    assert result["income_statement"].loc["Total Revenue"].iloc[0] == pytest.approx(120)
