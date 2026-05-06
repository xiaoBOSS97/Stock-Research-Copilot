"""Financial statement loading and normalization with yfinance."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf


STATEMENT_KEYS = ("income_statement", "balance_sheet", "cash_flow")


class FinancialDataError(RuntimeError):
    """Raised when financial statements cannot be loaded or normalized."""


def _clean_ticker(ticker: str) -> str:
    """Normalize a user-entered ticker symbol."""

    if not isinstance(ticker, str):
        raise TypeError("Ticker must be a string.")
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Ticker must not be empty.")
    return cleaned


def get_financial_statements(ticker: str) -> dict[str, pd.DataFrame]:
    """Load annual financial statements for a ticker from yfinance.

    Args:
        ticker: Public equity ticker symbol, for example ``AAPL``.

    Returns:
        A normalized dictionary with ``income_statement``, ``balance_sheet``,
        and ``cash_flow`` DataFrames.

    Raises:
        FinancialDataError: If yfinance fails or all statements are empty.
    """

    symbol = _clean_ticker(ticker)
    try:
        ticker_obj = yf.Ticker(symbol)
        raw = {
            "income_statement": ticker_obj.income_stmt,
            "balance_sheet": ticker_obj.balance_sheet,
            "cash_flow": ticker_obj.cashflow,
        }
    except Exception as exc:  # pragma: no cover - depends on network/client behavior
        raise FinancialDataError(f"Could not download financial statements for {symbol}: {exc}") from exc

    normalized = normalize_financials(raw)
    if all(statement.empty for statement in normalized.values()):
        raise FinancialDataError(f"No financial statements returned for {symbol}.")
    return normalized


def normalize_financials(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalize financial statement orientation, date columns, and numeric values.

    yfinance returns statement rows as financial concepts and columns as reporting
    dates. This project keeps that shape because ratio calculations can look up
    concepts by row name and compare periods across columns.
    """

    normalized: dict[str, pd.DataFrame] = {}
    for key in STATEMENT_KEYS:
        statement = raw.get(key, pd.DataFrame())
        if statement is None or statement.empty:
            normalized[key] = pd.DataFrame()
            continue

        frame = statement.copy()
        frame.index = frame.index.map(str)
        frame.columns = pd.to_datetime(frame.columns, errors="coerce")
        frame = frame.loc[:, frame.columns.notna()]
        frame = frame.sort_index(axis=1, ascending=False)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        normalized[key] = frame

    return normalized


def get_ttm_metrics(ticker: str) -> dict[str, float | None]:
    """Return basic trailing metrics useful for ratios and valuation inputs."""

    statements = get_financial_statements(ticker)
    income = statements["income_statement"]
    cash_flow = statements["cash_flow"]

    return {
        "eps": _latest_value(income, ("Diluted EPS", "Basic EPS")),
        "revenue": _latest_value(income, ("Total Revenue", "Operating Revenue")),
        "net_income": _latest_value(income, ("Net Income", "Net Income Common Stockholders")),
        "free_cash_flow": _latest_free_cash_flow(cash_flow),
    }


def _latest_value(statement: pd.DataFrame, row_names: tuple[str, ...]) -> float | None:
    """Return the newest numeric value for the first matching row name."""

    if statement.empty:
        return None
    for row_name in row_names:
        if row_name in statement.index:
            series = statement.loc[row_name].dropna()
            if not series.empty:
                return float(series.iloc[0])
    return None


def _latest_free_cash_flow(cash_flow: pd.DataFrame) -> float | None:
    """Return latest FCF, using reported FCF or OCF minus capex fallback."""

    reported = _latest_value(cash_flow, ("Free Cash Flow",))
    if reported is not None:
        return reported

    operating_cash_flow = _latest_value(
        cash_flow,
        ("Operating Cash Flow", "Total Cash From Operating Activities"),
    )
    capital_expenditure = _latest_value(
        cash_flow,
        ("Capital Expenditure", "Capital Expenditures", "CapitalExpenditures"),
    )
    if operating_cash_flow is None or capital_expenditure is None:
        return None
    return operating_cash_flow + capital_expenditure


def statements_to_dict(statements: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Convert statement DataFrames into plain dictionaries for serialization."""

    return {key: frame.to_dict() for key, frame in statements.items()}
