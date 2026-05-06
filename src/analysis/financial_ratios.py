"""Financial ratio calculations for normalized statement data."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _find_row(statement: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    """Return the first matching statement row for a list of possible labels."""

    if statement.empty:
        return None
    normalized_index = {str(index).lower(): index for index in statement.index}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized_index:
            return pd.to_numeric(statement.loc[normalized_index[key]], errors="coerce")
    return None


def _latest(series: pd.Series | None) -> float | None:
    """Return the latest non-null value from a statement row."""

    if series is None:
        return None
    values = series.dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    """Divide two values, returning ``None`` for unavailable or invalid inputs."""

    if numerator is None or denominator in (None, 0):
        return None
    result = numerator / denominator
    if not math.isfinite(result):
        return None
    return float(result)


def _pct_change_latest(series: pd.Series | None) -> float | None:
    """Return latest year-over-year growth from a newest-first series."""

    if series is None:
        return None
    values = series.dropna()
    if len(values) < 2:
        return None
    previous = float(values.iloc[1])
    if previous == 0:
        return None
    return float(values.iloc[0] / previous - 1)


def _cagr(series: pd.Series | None) -> float | None:
    """Return CAGR from newest to oldest value in a statement row."""

    if series is None:
        return None
    values = series.dropna()
    if len(values) < 2:
        return None
    start = float(values.iloc[-1])
    end = float(values.iloc[0])
    periods = len(values) - 1
    if start <= 0 or end <= 0 or periods <= 0:
        return None
    return float((end / start) ** (1 / periods) - 1)


def calculate_free_cash_flow(cash_flow: pd.DataFrame) -> pd.Series:
    """Calculate free cash flow as operating cash flow minus capital expenditure."""

    reported_fcf = _find_row(cash_flow, ("Free Cash Flow",))
    if reported_fcf is not None:
        return reported_fcf

    operating_cash_flow = _find_row(
        cash_flow,
        ("Operating Cash Flow", "Total Cash From Operating Activities"),
    )
    capital_expenditure = _find_row(
        cash_flow,
        ("Capital Expenditure", "Capital Expenditures", "CapitalExpenditures"),
    )
    if operating_cash_flow is None or capital_expenditure is None:
        raise ValueError("Cash flow statement must include operating cash flow and capital expenditure.")
    return operating_cash_flow + capital_expenditure


def calculate_growth_rates(income_statement: pd.DataFrame) -> dict[str, float | None]:
    """Calculate revenue and net income growth metrics."""

    revenue = _find_row(income_statement, ("Total Revenue", "Operating Revenue"))
    net_income = _find_row(
        income_statement,
        ("Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"),
    )

    return {
        "revenue_growth_yoy": _pct_change_latest(revenue),
        "net_income_growth_yoy": _pct_change_latest(net_income),
        "revenue_cagr": _cagr(revenue),
    }


def calculate_margins(
    income_statement: pd.DataFrame,
    cash_flow: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    """Calculate gross, operating, net, and free cash flow margins."""

    revenue = _latest(_find_row(income_statement, ("Total Revenue", "Operating Revenue")))
    gross_profit = _latest(_find_row(income_statement, ("Gross Profit",)))
    operating_income = _latest(_find_row(income_statement, ("Operating Income",)))
    net_income = _latest(
        _find_row(
            income_statement,
            ("Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"),
        )
    )

    fcf_margin = None
    if cash_flow is not None and not cash_flow.empty:
        try:
            fcf_margin = _safe_divide(_latest(calculate_free_cash_flow(cash_flow)), revenue)
        except ValueError:
            fcf_margin = None

    return {
        "gross_margin": _safe_divide(gross_profit, revenue),
        "operating_margin": _safe_divide(operating_income, revenue),
        "net_margin": _safe_divide(net_income, revenue),
        "fcf_margin": fcf_margin,
    }


def calculate_returns(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
) -> dict[str, float | None]:
    """Calculate return on equity, return on assets, and a simple ROIC proxy."""

    net_income = _latest(
        _find_row(
            income_statement,
            ("Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"),
        )
    )
    operating_income = _latest(_find_row(income_statement, ("Operating Income",)))
    tax_rate = _safe_divide(
        _latest(_find_row(income_statement, ("Tax Provision", "Income Tax Expense"))),
        _latest(_find_row(income_statement, ("Pretax Income", "Income Before Tax"))),
    )
    equity = _latest(
        _find_row(balance_sheet, ("Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"))
    )
    assets = _latest(_find_row(balance_sheet, ("Total Assets",)))
    debt = _latest(_find_row(balance_sheet, ("Total Debt", "Long Term Debt And Capital Lease Obligation")))
    cash = _latest(_find_row(balance_sheet, ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")))

    nopat = None
    if operating_income is not None:
        nopat = operating_income * (1 - tax_rate) if tax_rate is not None else operating_income
    invested_capital = None
    if debt is not None and equity is not None:
        invested_capital = debt + equity - (cash or 0)

    return {
        "roe": _safe_divide(net_income, equity),
        "roa": _safe_divide(net_income, assets),
        "roic": _safe_divide(nopat, invested_capital),
    }


def calculate_leverage(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
) -> dict[str, float | None]:
    """Calculate leverage and liquidity ratios."""

    total_debt = _latest(_find_row(balance_sheet, ("Total Debt", "Long Term Debt And Capital Lease Obligation")))
    equity = _latest(
        _find_row(balance_sheet, ("Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"))
    )
    current_assets = _latest(_find_row(balance_sheet, ("Current Assets", "Total Current Assets")))
    current_liabilities = _latest(_find_row(balance_sheet, ("Current Liabilities", "Total Current Liabilities")))
    operating_income = _latest(_find_row(income_statement, ("Operating Income",)))
    interest_expense = _latest(_find_row(income_statement, ("Interest Expense",)))
    if interest_expense is not None:
        interest_expense = abs(interest_expense)

    return {
        "debt_to_equity": _safe_divide(total_debt, equity),
        "current_ratio": _safe_divide(current_assets, current_liabilities),
        "interest_coverage": _safe_divide(operating_income, interest_expense),
    }


def calculate_financial_metrics(statements: dict[str, pd.DataFrame]) -> dict[str, float | None]:
    """Calculate all MVP financial metrics from normalized statements."""

    income_statement = statements.get("income_statement", pd.DataFrame())
    balance_sheet = statements.get("balance_sheet", pd.DataFrame())
    cash_flow = statements.get("cash_flow", pd.DataFrame())

    metrics: dict[str, float | None] = {}
    metrics.update(calculate_growth_rates(income_statement))
    metrics.update(calculate_margins(income_statement, cash_flow))
    metrics.update(calculate_returns(income_statement, balance_sheet))
    metrics.update(calculate_leverage(income_statement, balance_sheet))
    if cash_flow.empty:
        metrics["free_cash_flow"] = None
    else:
        try:
            metrics["free_cash_flow"] = _latest(calculate_free_cash_flow(cash_flow))
        except ValueError:
            metrics["free_cash_flow"] = None
    return metrics


def generate_financial_summary(metrics: dict[str, Any]) -> str:
    """Generate a rule-based financial summary using calculated metrics."""

    revenue_growth = metrics.get("revenue_growth_yoy")
    net_margin = metrics.get("net_margin")
    fcf_margin = metrics.get("fcf_margin")
    debt_to_equity = metrics.get("debt_to_equity")

    growth_text = "Revenue growth data is limited."
    if revenue_growth is not None:
        if revenue_growth > 0.15:
            growth_text = "Revenue growth appears strong on a year-over-year basis."
        elif revenue_growth > 0:
            growth_text = "Revenue is growing modestly year over year."
        else:
            growth_text = "Revenue declined year over year, which deserves closer review."

    profitability_text = "Profitability data is limited."
    if net_margin is not None:
        profitability_text = (
            "Net margin is positive, indicating the company is profitable."
            if net_margin > 0
            else "Net margin is negative, indicating profitability pressure."
        )

    cash_text = "Free cash flow data is limited."
    if fcf_margin is not None:
        cash_text = (
            "Free cash flow conversion is positive."
            if fcf_margin > 0
            else "Free cash flow conversion is negative."
        )

    leverage_text = "Balance sheet leverage data is limited."
    if debt_to_equity is not None:
        leverage_text = (
            "Debt-to-equity is elevated and should be monitored."
            if debt_to_equity > 1.5
            else "Debt-to-equity does not appear elevated from this simple ratio."
        )

    return " ".join([growth_text, profitability_text, cash_text, leverage_text])
