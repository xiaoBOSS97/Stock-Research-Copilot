"""Formatting helpers shared by dashboard and reports."""

from __future__ import annotations

from typing import Any

import pandas as pd


PERCENT_METRICS = {
    "annualized_volatility",
    "max_drawdown",
    "revenue_growth_yoy",
    "net_income_growth_yoy",
    "revenue_cagr",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "fcf_margin",
    "roe",
    "roa",
    "roic",
}
LARGE_CURRENCY_METRICS = {
    "marketCap",
    "market_cap",
    "revenue",
    "net_income",
    "free_cash_flow",
    "forward_revenue",
    "pe_target_price",
    "ps_target_price",
}
PRICE_METRICS = {
    "latest_close",
    "Close",
    "MA20",
    "MA50",
    "MA200",
    "target_price",
    "forward_eps",
    "eps",
    "pe_multiple",
    "ps_multiple",
}
LABEL_OVERRIDES = {
    "RSI": "RSI",
    "MA20": "MA20",
    "MA50": "MA50",
    "MA200": "MA200",
    "fcf_margin": "FCF Margin",
    "roe": "ROE",
    "roa": "ROA",
    "roic": "ROIC",
    "pe_multiple": "P/E Multiple",
    "ps_multiple": "P/S Multiple",
    "pe_weight": "P/E Weight",
    "ps_weight": "P/S Weight",
    "pe_target_price": "P/E Target Price",
    "ps_target_price": "P/S Target Price",
    "debt_to_equity": "Debt-to-Equity",
    "current_ratio": "Current Ratio",
    "interest_coverage": "Interest Coverage",
    "revenue_growth_yoy": "Revenue Growth YoY",
    "net_income_growth_yoy": "Net Income Growth YoY",
    "revenue_cagr": "Revenue CAGR",
}


def humanize_label(name: str) -> str:
    """Convert internal metric names into readable labels."""

    if name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[name]
    return name.replace("_", " ").replace("-", " ").title()


def format_value(value: Any, metric_name: str | None = None) -> str:
    """Format a value for dashboard tables and Markdown reports."""

    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)

    name = metric_name or ""
    numeric_value = float(value)
    if name in PERCENT_METRICS:
        return f"{numeric_value:.1%}"
    if name in LARGE_CURRENCY_METRICS:
        return format_large_number(numeric_value)
    if name in PRICE_METRICS:
        return f"{numeric_value:,.2f}"
    return f"{numeric_value:,.2f}"


def format_large_number(value: float) -> str:
    """Format large values with compact B/T suffixes."""

    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,.2f}"
