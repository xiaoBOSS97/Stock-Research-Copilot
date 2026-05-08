"""Financial statement loading and normalization with yfinance."""

from __future__ import annotations

from pathlib import Path
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


def get_financial_statements(
    ticker: str,
    *,
    cache: bool = False,
    cache_dir: str | Path = "data/processed",
    use_cache_fallback: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load annual financial statements for a ticker from yfinance.

    Args:
        ticker: Public equity ticker symbol, for example ``AAPL``.
        cache: Whether to persist normalized statements as CSV files.
        cache_dir: Directory for cached normalized statements.
        use_cache_fallback: Whether to load cached statements when live data is empty.

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
        if use_cache_fallback:
            cached = load_cached_financials(symbol, cache_dir=cache_dir)
            if any(not statement.empty for statement in cached.values()):
                return cached
        raise FinancialDataError(f"No financial statements returned for {symbol}.")
    if cache:
        save_financial_statements(normalized, symbol, output_dir=cache_dir)
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

    symbol = _clean_ticker(ticker)
    try:
        info: dict[str, Any] = yf.Ticker(symbol).info
    except Exception:  # pragma: no cover - depends on network/client behavior
        info = {}

    statements = get_financial_statements(symbol, cache=True)
    income = statements["income_statement"]
    cash_flow = statements["cash_flow"]

    return {
        "eps": _first_numeric(
            info.get("forwardEps"),
            info.get("trailingEps"),
            _latest_value(income, ("Diluted EPS", "Basic EPS")),
        ),
        "revenue": _first_numeric(
            info.get("totalRevenue"),
            _latest_value(income, ("Total Revenue", "Operating Revenue")),
        ),
        "net_income": _latest_value(income, ("Net Income", "Net Income Common Stockholders")),
        "free_cash_flow": _first_numeric(info.get("freeCashflow"), _latest_free_cash_flow(cash_flow)),
    }


def _first_numeric(*values: Any) -> float | None:
    """Return the first numeric value from a list of provider fields."""

    for value in values:
        if value is None:
            continue
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            continue
        return float(numeric)
    return None


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


def save_financial_statements(
    statements: dict[str, pd.DataFrame],
    ticker: str,
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Save normalized financial statements as CSV files."""

    symbol = _clean_ticker(ticker)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for key in STATEMENT_KEYS:
        statement = statements.get(key, pd.DataFrame())
        if statement.empty:
            continue
        path = target_dir / f"{symbol}_{key}.csv"
        statement.to_csv(path)
        paths[key] = path
    return paths


def load_cached_financials(
    ticker: str,
    cache_dir: str | Path = "data/processed",
) -> dict[str, pd.DataFrame]:
    """Load cached normalized financial statements when available."""

    symbol = _clean_ticker(ticker)
    source_dir = Path(cache_dir)
    statements: dict[str, pd.DataFrame] = {}
    for key in STATEMENT_KEYS:
        path = source_dir / f"{symbol}_{key}.csv"
        if not path.exists():
            statements[key] = pd.DataFrame()
            continue
        frame = pd.read_csv(path, index_col=0)
        frame.columns = pd.to_datetime(frame.columns, errors="coerce")
        frame = frame.loc[:, frame.columns.notna()]
        frame = frame.apply(pd.to_numeric, errors="coerce")
        statements[key] = frame
    return statements
