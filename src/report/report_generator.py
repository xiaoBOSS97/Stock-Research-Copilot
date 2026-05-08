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
from src.utils.formatting import calculate_upside_downside, format_value, humanize_label


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
    peer_text = peer_comparison or ""

    return {
        "ticker": symbol,
        "company_name": profile.get("name") or symbol,
        "generated_at": generated_time.isoformat(timespec="seconds"),
        "one_line_summary": _one_line_summary(symbol, technical_metrics, valuation_text),
        "executive_summary": _executive_summary(symbol, technical_metrics, metrics, valuation),
        "company_profile": _format_company_profile(profile),
        "price_analysis": _format_price_analysis(technical_metrics, technical_summary),
        "financial_analysis": _format_financial_analysis(metrics, financial_text),
        "valuation_analysis": valuation_text,
        "valuation_assumptions": _format_valuation_assumptions(
            valuation,
            current_price=technical_metrics["latest_close"],
        ),
        "has_peer_comparison": bool(peer_text.strip()),
        "peer_comparison": peer_text,
        "scenario_table": _format_markdown_table(valuation),
        "data_quality_notes": _data_quality_notes(price_history, metrics, valuation),
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
        shares_outstanding=profile.get("sharesOutstanding"),
        base_pe_multiple=_first_positive(profile.get("forwardPE"), profile.get("trailingPE")) or 22.0,
        base_ps_multiple=_first_positive(profile.get("priceToSalesTrailing12Months")),
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
    shares_outstanding: float | None = None,
    base_pe_multiple: float = 22.0,
    base_ps_multiple: float | None = None,
) -> tuple[pd.DataFrame, str]:
    """Build a default valuation table from yfinance metrics when available."""

    try:
        ttm = get_ttm_metrics(ticker)
    except FinancialDataError:
        ttm = {}

    base_eps = ttm.get("eps") or _fallback_eps(current_price)
    base_revenue = ttm.get("revenue")
    base_free_cash_flow = ttm.get("free_cash_flow")
    shares_outstanding = shares_outstanding or _estimate_shares_outstanding(market_cap, current_price)
    if base_ps_multiple is None and market_cap is not None and base_revenue:
        base_ps_multiple = market_cap / base_revenue
    assumptions = build_scenarios(
        base_forward_eps=base_eps,
        base_pe_multiple=base_pe_multiple,
        base_forward_revenue=base_revenue,
        base_ps_multiple=base_ps_multiple if base_revenue else None,
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


def _first_positive(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric) or float(numeric) <= 0:
            continue
        return float(numeric)
    return None


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


def _executive_summary(
    ticker: str,
    technical_metrics: dict[str, Any],
    financial_metrics: dict[str, float | None],
    valuation_table: pd.DataFrame,
) -> str:
    """Build a concise analyst-style summary without making recommendations."""

    trend_text = (
        "constructive versus the long-term moving average"
        if technical_metrics["trend"] == "above_ma200"
        else "weaker versus the long-term moving average"
    )
    revenue_growth = financial_metrics.get("revenue_growth_yoy")
    net_margin = financial_metrics.get("net_margin")
    fcf_margin = financial_metrics.get("fcf_margin")
    valuation_range = _valuation_range_text(valuation_table, technical_metrics["latest_close"])

    return "\n".join(
        [
            f"- **Market setup:** {ticker} screens as {trend_text}; RSI is {format_value(technical_metrics['RSI'], 'RSI')} and annualized volatility is {format_value(technical_metrics['annualized_volatility'], 'annualized_volatility')}.",
            f"- **Fundamentals:** Revenue growth is {format_value(revenue_growth, 'revenue_growth_yoy')}, net margin is {format_value(net_margin, 'net_margin')}, and FCF margin is {format_value(fcf_margin, 'fcf_margin')}.",
            f"- **Valuation frame:** {valuation_range}",
        ]
    )


def _valuation_range_text(valuation_table: pd.DataFrame, current_price: float | None) -> str:
    if valuation_table.empty or "target_price" not in valuation_table:
        return "Valuation scenarios are unavailable because required inputs are missing."

    prices = pd.to_numeric(valuation_table["target_price"], errors="coerce").dropna()
    if prices.empty:
        return "Valuation scenarios are unavailable because required inputs are missing."

    low = float(prices.min())
    high = float(prices.max())
    base_price = _scenario_target_price(valuation_table, "Base")
    base_text = "N/A" if base_price is None else format_value(base_price, "target_price")
    current_text = "N/A" if current_price is None else format_value(current_price, "latest_close")
    return (
        f"The scenario range is {format_value(low, 'target_price')} to {format_value(high, 'target_price')}; "
        f"the base case is {base_text} versus current price of {current_text}."
    )


def _scenario_target_price(valuation_table: pd.DataFrame, scenario: str) -> float | None:
    if valuation_table.empty or "scenario" not in valuation_table or "target_price" not in valuation_table:
        return None
    rows = valuation_table[valuation_table["scenario"] == scenario]
    if rows.empty:
        return None
    value = pd.to_numeric(rows["target_price"], errors="coerce").dropna()
    if value.empty:
        return None
    return float(value.iloc[0])


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


def _format_valuation_assumptions(frame: pd.DataFrame, current_price: float | None = None) -> str:
    if frame.empty or "assumed_inputs" not in frame.columns:
        return "No valuation assumptions were supplied."

    rows = []
    show_status = "status" in frame.columns and not (frame["status"] == "ok").all()
    for _, row in frame.iterrows():
        display_row = {
            "Scenario": row.get("scenario", "N/A"),
            "Method": row.get("method", "N/A"),
            "Target Price": format_value(row.get("target_price"), "target_price"),
            "Upside/Downside": format_value(
                calculate_upside_downside(row.get("target_price"), current_price),
                "upside_downside",
            ),
            "Key Assumptions": _compact_assumptions(row.get("assumed_inputs")),
        }
        if show_status:
            display_row["Status"] = row.get("status", "N/A")
        rows.append(display_row)
    return _rows_to_markdown(rows)


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
        ]
    )


def _data_quality_notes(
    price_history: pd.DataFrame,
    financial_metrics: dict[str, float | None],
    valuation_table: pd.DataFrame,
) -> str:
    """Describe report input coverage and known caveats."""

    price_start = price_history.index.min().date().isoformat() if not price_history.empty else "N/A"
    price_end = price_history.index.max().date().isoformat() if not price_history.empty else "N/A"
    available_metrics = sum(value is not None and not pd.isna(value) for value in financial_metrics.values())
    total_metrics = len(financial_metrics)
    valuation_ok = 0
    if not valuation_table.empty and "status" in valuation_table:
        valuation_ok = int((valuation_table["status"] == "ok").sum())

    return "\n".join(
        [
            f"- Price history covers {len(price_history)} rows from {price_start} to {price_end}.",
            f"- Financial metrics available: {available_metrics}/{total_metrics if total_metrics else 0}.",
            f"- Valuation scenarios calculated successfully: {valuation_ok}/{len(valuation_table)}.",
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
