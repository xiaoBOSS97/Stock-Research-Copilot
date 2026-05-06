"""Technical analysis helpers for daily equity price data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MA_WINDOWS = (20, 50, 200)
TRADING_DAYS_PER_YEAR = 252


def _require_close(price_data: pd.DataFrame) -> pd.Series:
    """Return a numeric close series after validating the input frame."""

    if price_data.empty:
        raise ValueError("Price data must not be empty.")
    if "Close" not in price_data.columns:
        raise ValueError("Price data must include a 'Close' column.")
    close = pd.to_numeric(price_data["Close"], errors="coerce")
    if close.dropna().empty:
        raise ValueError("Price data must include at least one numeric close price.")
    return close


def add_moving_averages(
    price_data: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_MA_WINDOWS,
) -> pd.DataFrame:
    """Return price data with simple moving average columns added."""

    close = _require_close(price_data)
    result = price_data.copy()
    for window in windows:
        if window <= 0:
            raise ValueError("Moving average windows must be positive integers.")
        result[f"MA{window}"] = close.rolling(window=window, min_periods=1).mean()
    return result


def calculate_rsi(price_data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate relative strength index for close prices.

    RSI is returned as a 0-100 series. Early rows with insufficient history are
    filled with a neutral value of 50 to keep dashboards and reports stable.
    """

    if window <= 0:
        raise ValueError("RSI window must be a positive integer.")

    close = _require_close(price_data)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    no_losses = avg_loss.eq(0)
    no_gains = avg_gain.eq(0)
    rsi = rsi.mask(no_losses & ~no_gains, 100)
    rsi = rsi.mask(no_gains & ~no_losses, 0)
    rsi = rsi.mask(no_gains & no_losses, 50)
    rsi = rsi.fillna(50).clip(0, 100)
    rsi.name = "RSI"
    return rsi


def add_rsi(price_data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Return price data with an RSI column added."""

    result = price_data.copy()
    result["RSI"] = calculate_rsi(result, window=window)
    return result


def calculate_volatility(price_data: pd.DataFrame, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calculate annualized volatility from daily close-to-close returns."""

    if trading_days <= 0:
        raise ValueError("Trading days must be a positive integer.")

    close = _require_close(price_data)
    returns = close.pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * np.sqrt(trading_days))


def calculate_max_drawdown(price_data: pd.DataFrame) -> float:
    """Calculate maximum drawdown as a non-positive decimal value."""

    close = _require_close(price_data).dropna()
    running_max = close.cummax()
    drawdown = close / running_max - 1
    return float(drawdown.min())


def add_technical_indicators(
    price_data: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_MA_WINDOWS,
    rsi_window: int = 14,
) -> pd.DataFrame:
    """Return price data with MVP technical indicators added."""

    result = add_moving_averages(price_data, windows=windows)
    result["RSI"] = calculate_rsi(result, window=rsi_window)
    result["annualized_volatility"] = calculate_volatility(result)
    result["max_drawdown"] = calculate_max_drawdown(result)
    return result


def calculate_technical_metrics(price_data: pd.DataFrame) -> dict[str, Any]:
    """Calculate latest technical metrics for dashboard cards and reports.

    Returns:
        A dictionary containing latest close, MA20/50/200, RSI, annualized
        volatility, max drawdown, trend label, and momentum label.
    """

    enriched = add_technical_indicators(price_data)
    latest = enriched.iloc[-1]
    close = float(latest["Close"])
    ma200 = float(latest["MA200"])
    rsi = float(latest["RSI"])

    return {
        "latest_close": close,
        "MA20": float(latest["MA20"]),
        "MA50": float(latest["MA50"]),
        "MA200": ma200,
        "RSI": rsi,
        "annualized_volatility": float(latest["annualized_volatility"]),
        "max_drawdown": float(latest["max_drawdown"]),
        "trend": "above_ma200" if close >= ma200 else "below_ma200",
        "momentum": _momentum_label(rsi),
    }


def _momentum_label(rsi: float) -> str:
    if rsi > 70:
        return "overheated"
    if rsi < 30:
        return "oversold"
    return "neutral"


def generate_technical_summary(price_data: pd.DataFrame) -> str:
    """Generate a rule-based technical summary for reports."""

    metrics = calculate_technical_metrics(price_data)

    trend = "above" if metrics["trend"] == "above_ma200" else "below"
    return (
        f"The latest close is {trend} the 200-day moving average, while RSI is "
        f"{metrics['RSI']:.1f}, indicating {metrics['momentum']} short-term momentum. "
        "Technical indicators "
        "are market behavior signals and do not replace fundamental analysis."
    )
