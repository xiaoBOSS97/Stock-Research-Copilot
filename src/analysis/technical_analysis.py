"""Technical analysis helpers for daily equity price data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_close(price_data: pd.DataFrame) -> pd.Series:
    if price_data.empty:
        raise ValueError("Price data must not be empty.")
    if "Close" not in price_data.columns:
        raise ValueError("Price data must include a 'Close' column.")
    close = pd.to_numeric(price_data["Close"], errors="coerce")
    if close.dropna().empty:
        raise ValueError("Price data must include at least one numeric close price.")
    return close


def add_moving_averages(price_data: pd.DataFrame, windows: tuple[int, ...] = (20, 50, 200)) -> pd.DataFrame:
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

    rsi = rsi.where(avg_loss.ne(0), 100)
    rsi = rsi.where(avg_gain.ne(0), 0)
    rsi = rsi.fillna(50).clip(0, 100)
    rsi.name = "RSI"
    return rsi


def add_rsi(price_data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Return price data with an RSI column added."""

    result = price_data.copy()
    result["RSI"] = calculate_rsi(result, window=window)
    return result


def calculate_volatility(price_data: pd.DataFrame, trading_days: int = 252) -> float:
    """Calculate annualized volatility from daily close-to-close returns."""

    close = _require_close(price_data)
    returns = close.pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * np.sqrt(trading_days))


def calculate_max_drawdown(price_data: pd.DataFrame) -> float:
    """Calculate maximum drawdown as a non-positive decimal value."""

    close = _require_close(price_data)
    running_max = close.cummax()
    drawdown = close / running_max - 1
    return float(drawdown.min())


def add_technical_indicators(price_data: pd.DataFrame) -> pd.DataFrame:
    """Return price data with MVP technical indicators added."""

    result = add_moving_averages(price_data)
    result["RSI"] = calculate_rsi(result)
    result["annualized_volatility"] = calculate_volatility(result)
    result["max_drawdown"] = calculate_max_drawdown(result)
    return result


def generate_technical_summary(price_data: pd.DataFrame) -> str:
    """Generate a rule-based technical summary for reports."""

    enriched = add_technical_indicators(price_data)
    latest = enriched.iloc[-1]
    close = latest["Close"]
    ma200 = latest["MA200"]
    rsi = latest["RSI"]

    trend = "above" if close >= ma200 else "below"
    momentum = "overheated" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
    return (
        f"The latest close is {trend} the 200-day moving average, while RSI is "
        f"{rsi:.1f}, indicating {momentum} short-term momentum. Technical indicators "
        "are market behavior signals and do not replace fundamental analysis."
    )
