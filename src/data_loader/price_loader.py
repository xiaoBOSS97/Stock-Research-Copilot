"""Price and company profile loading with yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf


class PriceDataError(RuntimeError):
    """Raised when price or profile data cannot be loaded clearly."""


@dataclass(frozen=True)
class CompanyProfile:
    """Basic company profile fields used by the dashboard and reports."""

    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: int | None = None
    currency: str | None = None


def _clean_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Ticker must not be empty.")
    return cleaned


def get_price_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download historical OHLCV data for a ticker.

    Args:
        ticker: Public equity ticker symbol, for example ``AAPL``.
        period: yfinance period value, such as ``1y``, ``3y``, ``5y`` or ``max``.
        interval: yfinance interval value, such as ``1d``.

    Returns:
        A DataFrame indexed by date with standard OHLCV columns.

    Raises:
        PriceDataError: If yfinance returns no usable rows or the request fails.
    """

    symbol = _clean_ticker(ticker)
    try:
        history = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    except Exception as exc:  # pragma: no cover - depends on network/client behavior
        raise PriceDataError(f"Could not download price history for {symbol}: {exc}") from exc

    if history.empty:
        raise PriceDataError(f"No price history returned for {symbol}.")

    history = history.rename_axis("Date").sort_index()
    expected_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [column for column in expected_columns if column not in history.columns]
    if missing:
        raise PriceDataError(f"Price history for {symbol} is missing columns: {missing}.")

    return history[expected_columns].dropna(subset=["Close"])


def get_company_profile(ticker: str) -> CompanyProfile:
    """Fetch a small company profile from yfinance metadata."""

    symbol = _clean_ticker(ticker)
    try:
        info: dict[str, Any] = yf.Ticker(symbol).info
    except Exception as exc:  # pragma: no cover - depends on network/client behavior
        raise PriceDataError(f"Could not download company profile for {symbol}: {exc}") from exc

    if not info:
        raise PriceDataError(f"No company profile returned for {symbol}.")

    return CompanyProfile(
        ticker=symbol,
        name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=info.get("marketCap"),
        currency=info.get("currency"),
    )


def validate_ticker(ticker: str) -> bool:
    """Return whether a ticker appears to have available price data."""

    try:
        history = get_price_history(ticker=ticker, period="5d", interval="1d")
    except (ValueError, PriceDataError):
        return False
    return not history.empty
