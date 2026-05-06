"""Price and company profile loading with yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


EXPECTED_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


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

    def to_dict(self) -> dict[str, Any]:
        """Return the public profile shape expected by app and report layers."""

        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "industry": self.industry,
            "marketCap": self.market_cap,
            "currency": self.currency,
        }


def _clean_ticker(ticker: str) -> str:
    """Normalize a user-entered ticker symbol."""

    if not isinstance(ticker, str):
        raise TypeError("Ticker must be a string.")
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Ticker must not be empty.")
    return cleaned


def _normalize_price_history(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance history output into a stable OHLCV DataFrame."""

    if history.empty:
        raise PriceDataError(f"No price history returned for {ticker}.")

    missing = [column for column in EXPECTED_PRICE_COLUMNS if column not in history.columns]
    if missing:
        raise PriceDataError(f"Price history for {ticker} is missing columns: {missing}.")

    normalized = history.loc[:, list(EXPECTED_PRICE_COLUMNS)].copy()
    normalized.index = pd.to_datetime(normalized.index)
    if getattr(normalized.index, "tz", None) is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized = normalized.rename_axis("Date").sort_index()

    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    normalized[numeric_columns] = normalized[numeric_columns].apply(pd.to_numeric, errors="coerce")
    normalized = normalized.dropna(subset=["Close"])
    if normalized.empty:
        raise PriceDataError(f"Price history for {ticker} does not contain usable close prices.")

    return normalized


def save_price_history(
    price_history: pd.DataFrame,
    ticker: str,
    period: str,
    interval: str,
    output_dir: str | Path = "data/raw",
) -> Path:
    """Save normalized price history to a CSV file and return the path."""

    symbol = _clean_ticker(ticker)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{symbol}_{period}_{interval}_prices.csv"
    price_history.to_csv(target_path)
    return target_path


def get_price_history(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    *,
    cache: bool = False,
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Download historical OHLCV data for a ticker.

    Args:
        ticker: Public equity ticker symbol, for example ``AAPL``.
        period: yfinance period value, such as ``1y``, ``3y``, ``5y`` or ``max``.
        interval: yfinance interval value, such as ``1d``.
        cache: Whether to save the normalized data as a CSV under ``cache_dir``.
        cache_dir: Directory used when ``cache`` is enabled.

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

    normalized = _normalize_price_history(history, symbol)
    if cache:
        save_price_history(normalized, symbol, period, interval, output_dir=cache_dir)

    return normalized


def get_company_profile(ticker: str) -> dict[str, Any]:
    """Fetch a small company profile from yfinance metadata."""

    symbol = _clean_ticker(ticker)
    try:
        info: dict[str, Any] = yf.Ticker(symbol).info
    except Exception as exc:  # pragma: no cover - depends on network/client behavior
        raise PriceDataError(f"Could not download company profile for {symbol}: {exc}") from exc

    if not info:
        raise PriceDataError(f"No company profile returned for {symbol}.")

    profile = CompanyProfile(
        ticker=symbol,
        name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=info.get("marketCap"),
        currency=info.get("currency"),
    )
    return profile.to_dict()


def validate_ticker(ticker: str) -> bool:
    """Return whether a ticker appears to have available price data."""

    try:
        history = get_price_history(ticker=ticker, period="5d", interval="1d")
    except (TypeError, ValueError, PriceDataError):
        return False
    return not history.empty
