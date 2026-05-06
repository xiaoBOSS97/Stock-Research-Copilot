"""Markdown equity report generation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.analysis.financial_ratios import calculate_financial_metrics, generate_financial_summary
from src.analysis.technical_analysis import calculate_technical_metrics, generate_technical_summary
from src.analysis.valuation import (
    blended_valuation_with_dcf,
    build_dcf_scenarios,
    build_scenarios,
    summarize_valuation,
)
from src.data_loader.financial_loader import FinancialDataError, get_financial_statements, get_ttm_metrics
from src.data_loader.price_loader import get_company_profile, get_price_history
from src.utils.formatting import format_value, humanize_label


DISCLAIMER = (
    "This project is for educational and research purposes only. It does not provide financial "
    "advice, investment recommendations, or trading signals. The analysis is based on public data "
    "and model assumptions, which may be incomplete or inaccurate."
)
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
DEFAULT_TEMPLATE = "equity_report.md.j2"


def build_report_context(
    ticker: str,
    price_history: pd.DataFrame,
    company_profile: dict[str, Any] | None = None,
    financial_metrics: dict[str, float | None] | None = None,
    financial_summary: str | None = None,
    valuation_table: pd.DataFrame | None = None,
    valuation_summary: str | None = None,
    peer_comparison: str | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the Jinja context for a Markdown equity report."""

    symbol = ticker.strip().upper()
    profile = company_profile or {"ticker": symbol, "name": symbol}
    technical_metrics = calculate_technical_metrics(price_history)
    technical_summary = generate_technical_summary(price_history)

    generated_time = generated_at or datetime.now(UTC)
    valuation = valuation_table if valuation_table is not None else _empty_scenario_table()
    valuation_text = valuation_summary or summarize_valuation(
        valuation,
        current_price=technical_metrics["latest_close"],
    )
    metrics = financial_metrics or {}
    financial_text = financial_summary or generate_financial_summary(metrics)

    return {
        "ticker": symbol,
        "company_name": profile.get("name") or symbol,
        "generated_at": generated_time.isoformat(timespec="seconds"),
        "one_line_summary": _one_line_summary(symbol, technical_metrics, valuation_text),
        "company_profile": _format_company_profile(profile),
        "price_analysis": _format_price_analysis(technical_metrics, technical_summary),
        "financial_analysis": _format_financial_analysis(metrics, financial_text),
        "valuation_analysis": valuation_text,
        "peer_comparison": peer_comparison or "No peer comparison table was supplied for this report.",
        "scenario_table": _format_markdown_table(valuation),
        "risk_factors": _default_risk_factors(),
        "watchlist": _default_watchlist(),
        "sources_and_disclaimer": _sources_and_disclaimer(),
        "technical_metrics": technical_metrics,
        "financial_metrics": metrics,
        "valuation_table": valuation,
    }


def render_markdown_report(
    context: dict[str, Any],
    template_name: str = DEFAULT_TEMPLATE,
    template_dir: str | Path = TEMPLATE_DIR,
) -> str:
    """Render a Markdown report from a Jinja template context."""

    environment = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(disabled_extensions=("md", "j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template(template_name)
    return template.render(**context).strip() + "\n"


def save_report(markdown: str, ticker: str, output_dir: str | Path = "data/reports") -> Path:
    """Save a Markdown report to ``data/reports/{ticker}_report.md``."""

    symbol = ticker.strip().upper()
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{symbol}_report.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def generate_report_for_ticker(
    ticker: str = "AAPL",
    period: str = "5y",
    interval: str = "1d",
    output_dir: str | Path = "data/reports",
) -> Path:
    """Run the MVP pipeline and save a Markdown report for a ticker."""

    symbol = ticker.strip().upper()
    price_history = get_price_history(symbol, period=period, interval=interval, cache=True)
    profile = get_company_profile(symbol)
    technical_metrics = calculate_technical_metrics(price_history)

    financial_metrics: dict[str, float | None]
    financial_summary: str
    try:
        statements = get_financial_statements(symbol, cache=True)
        financial_metrics = calculate_financial_metrics(statements)
        financial_summary = generate_financial_summary(financial_metrics)
    except FinancialDataError as exc:
        financial_metrics = {}
        financial_summary = f"Financial statement data is currently unavailable: {exc}"

    valuation_table, valuation_summary = _build_default_valuation(
        symbol,
        technical_metrics["latest_close"],
        market_cap=profile.get("marketCap"),
    )
    context = build_report_context(
        ticker=symbol,
        price_history=price_history,
        company_profile=profile,
        financial_metrics=financial_metrics,
        financial_summary=financial_summary,
        valuation_table=valuation_table,
        valuation_summary=valuation_summary,
    )
    markdown = render_markdown_report(context)
    return save_report(markdown, symbol, output_dir=output_dir)


def _build_default_valuation(
    ticker: str,
    current_price: float,
    market_cap: float | None = None,
) -> tuple[pd.DataFrame, str]:
    """Build a default valuation table from yfinance metrics when available."""

    try:
        ttm = get_ttm_metrics(ticker)
    except FinancialDataError:
        ttm = {}

    base_eps = ttm.get("eps") or _fallback_eps(current_price)
    base_revenue = ttm.get("revenue")
    base_free_cash_flow = ttm.get("free_cash_flow")
    shares_outstanding = _estimate_shares_outstanding(market_cap, current_price)
    assumptions = build_scenarios(
        base_forward_eps=base_eps,
        base_pe_multiple=22.0,
        base_forward_revenue=base_revenue,
        base_ps_multiple=6.0 if base_revenue else None,
        shares_outstanding=shares_outstanding,
    )
    dcf_assumptions = build_dcf_scenarios(
        base_free_cash_flow=base_free_cash_flow,
        base_growth_rate=0.05,
        discount_rate=0.10,
        terminal_growth_rate=0.025,
        net_debt=0.0,
        shares_outstanding=shares_outstanding,
    )
    valuation = blended_valuation_with_dcf(assumptions, dcf_assumptions=dcf_assumptions)
    return valuation, summarize_valuation(valuation, current_price=current_price)


def _estimate_shares_outstanding(market_cap: float | None, current_price: float) -> float | None:
    if market_cap is None or current_price <= 0:
        return None
    return float(market_cap / current_price)


def _fallback_eps(current_price: float) -> float:
    """Use a simple placeholder EPS assumption when financial data is missing."""

    return current_price / 25


def _one_line_summary(
    ticker: str,
    technical_metrics: dict[str, Any],
    valuation_summary: str,
) -> str:
    trend = (
        "above the 200-day moving average"
        if technical_metrics["trend"] == "above_ma200"
        else "below the 200-day moving average"
    )
    return (
        f"{ticker} is trading {trend} with "
        f"{technical_metrics['momentum']} short-term momentum. {valuation_summary}"
    )


def _format_company_profile(profile: dict[str, Any]) -> str:
    rows = [
        ("Ticker", profile.get("ticker")),
        ("Name", profile.get("name")),
        ("Sector", profile.get("sector")),
        ("Industry", profile.get("industry")),
        ("Market Cap", format_value(profile.get("marketCap"), "marketCap")),
        ("Currency", profile.get("currency")),
    ]
    return "\n".join(f"- {label}: {value if value is not None else 'N/A'}" for label, value in rows)


def _format_price_analysis(metrics: dict[str, Any], summary: str) -> str:
    rows = [
        {"Metric": "Latest Close", "Value": format_value(metrics["latest_close"], "latest_close")},
        {"Metric": "MA20", "Value": format_value(metrics["MA20"], "MA20")},
        {"Metric": "MA50", "Value": format_value(metrics["MA50"], "MA50")},
        {"Metric": "MA200", "Value": format_value(metrics["MA200"], "MA200")},
        {"Metric": "RSI", "Value": format_value(metrics["RSI"], "RSI")},
        {
            "Metric": "Annualized Volatility",
            "Value": format_value(metrics["annualized_volatility"], "annualized_volatility"),
        },
        {"Metric": "Max Drawdown", "Value": format_value(metrics["max_drawdown"], "max_drawdown")},
    ]
    return f"{summary}\n\n{_rows_to_markdown(rows)}"


def _format_financial_analysis(metrics: dict[str, float | None], summary: str) -> str:
    if not metrics:
        return summary

    rows = [
        {"Metric": humanize_label(key), "Value": format_value(value, key)}
        for key, value in metrics.items()
    ]
    return f"{summary}\n\n{_rows_to_markdown(rows)}"


def _format_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "Valuation scenarios are unavailable."
    display = frame.copy()
    if "assumed_inputs" in display.columns:
        display["assumed_inputs"] = display["assumed_inputs"].map(_compact_assumptions)
    if "target_price" in display.columns:
        display["target_price"] = display["target_price"].map(lambda value: format_value(value, "target_price"))
    return _rows_to_markdown(display.to_dict(orient="records"))


def format_dataframe_markdown(frame: pd.DataFrame) -> str:
    """Format a DataFrame as a lightweight Markdown table."""

    if frame.empty:
        return "N/A"
    return _rows_to_markdown(frame.to_dict(orient="records"))


def _rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    """Render dictionaries as a simple GitHub-flavored Markdown table."""

    if not rows:
        return "N/A"
    headers = list(rows[0].keys())
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_escape_table_cell(row.get(header_name)) for header_name in headers) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _escape_table_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _compact_assumptions(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)
    parts = []
    for key, item in value.items():
        if item is None:
            continue
        parts.append(f"{humanize_label(key)}={format_value(item, key)}")
    return ", ".join(parts) if parts else "N/A"


def _empty_scenario_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": ["Bear", "Base", "Bull"],
            "method": ["blended", "blended", "blended"],
            "assumed_inputs": [{}, {}, {}],
            "target_price": [None, None, None],
            "status": ["insufficient_data", "insufficient_data", "insufficient_data"],
        }
    )


def _default_risk_factors() -> str:
    return "\n".join(
        [
            "- Fundamental risk: revenue growth, margins, or cash flow may deteriorate.",
            "- Valuation risk: market multiples can compress even if fundamentals remain stable.",
            "- Market risk: macro conditions, interest rates, and sentiment can affect price behavior.",
            "- Data quality risk: public data sources may be delayed, incomplete, or inaccurate.",
        ]
    )


def _default_watchlist() -> str:
    return "\n".join(
        [
            "- Upcoming earnings and management guidance.",
            "- Revenue growth and margin trend changes.",
            "- Free cash flow conversion.",
            "- Valuation multiple changes versus peers and history.",
        ]
    )


def _sources_and_disclaimer() -> str:
    return "\n".join(
        [
            "- Price data: Yahoo Finance via yfinance.",
            "- Financial data: Yahoo Finance via yfinance when available.",
            f"- Disclaimer: {DISCLAIMER}",
        ]
    )


def main() -> None:
    """CLI entrypoint for report generation."""

    parser = argparse.ArgumentParser(description="Generate a Markdown equity research report.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--out", default=None, help="Optional explicit Markdown output path.")
    args = parser.parse_args()

    output_path = generate_report_for_ticker(args.ticker, period=args.period, interval=args.interval)
    if args.out:
        requested_path = Path(args.out)
        requested_path.parent.mkdir(parents=True, exist_ok=True)
        requested_path.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")
        output_path = requested_path
    print(output_path)


if __name__ == "__main__":
    main()
