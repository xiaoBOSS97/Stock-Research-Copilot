"""Streamlit dashboard for Stock Research Copilot."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.financial_ratios import calculate_financial_metrics, generate_financial_summary
from src.analysis.technical_analysis import add_technical_indicators, calculate_technical_metrics
from src.analysis.valuation import (
    blended_valuation,
    build_scenarios,
    estimate_by_pe,
    estimate_by_ps,
    summarize_valuation,
)
from src.data_loader.financial_loader import FinancialDataError, get_financial_statements
from src.data_loader.price_loader import PriceDataError, get_company_profile, get_price_history
from src.report.report_generator import (
    DISCLAIMER,
    build_report_context,
    format_dataframe_markdown,
    render_markdown_report,
    save_report,
)
from src.utils.formatting import format_value, humanize_label


CONFIG_DIR = PROJECT_ROOT / "config"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
PERIOD_OPTIONS = {"1Y": "1y", "3Y": "3y", "5Y": "5y", "Max": "max"}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty mapping when unavailable."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, period: str, interval: str) -> tuple[pd.DataFrame | None, str | None]:
    """Load price data from yfinance, falling back to cached CSV when available."""

    symbol = ticker.strip().upper()
    try:
        return get_price_history(symbol, period=period, interval=interval, cache=True), None
    except (PriceDataError, ValueError, TypeError) as exc:
        cached_path = RAW_DATA_DIR / f"{symbol}_{period}_{interval}_prices.csv"
        if cached_path.exists():
            frame = pd.read_csv(cached_path, index_col="Date", parse_dates=True)
            return frame, f"Using cached price data because live download failed: {exc}"
        return None, str(exc)


@st.cache_data(show_spinner=False)
def load_profile(ticker: str) -> tuple[dict[str, Any], str | None]:
    """Load a company profile with a compact fallback."""

    symbol = ticker.strip().upper()
    try:
        return get_company_profile(symbol), None
    except (PriceDataError, ValueError, TypeError) as exc:
        return {"ticker": symbol, "name": symbol}, str(exc)


@st.cache_data(show_spinner=False)
def load_financial_data(ticker: str) -> tuple[dict[str, pd.DataFrame], dict[str, float | None], str]:
    """Load financial statements and metrics, returning empty data on failure."""

    try:
        statements = get_financial_statements(ticker, cache=True)
        metrics = calculate_financial_metrics(statements)
        return statements, metrics, generate_financial_summary(metrics)
    except (FinancialDataError, ValueError, TypeError) as exc:
        return {}, {}, f"Financial statement data is currently unavailable: {exc}"


def parse_peer_input(raw_text: str) -> list[str]:
    """Parse comma-separated peer ticker input."""

    return [item.strip().upper() for item in raw_text.split(",") if item.strip()]


def build_price_chart(price_data: pd.DataFrame) -> go.Figure:
    """Build close price and moving average chart."""

    enriched = add_technical_indicators(price_data)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=enriched.index, y=enriched["Close"], name="Close", mode="lines"))
    for column in ("MA20", "MA50", "MA200"):
        figure.add_trace(go.Scatter(x=enriched.index, y=enriched[column], name=column, mode="lines"))
    figure.update_layout(
        height=460,
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        legend={"orientation": "h"},
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return figure


def build_financial_charts(statements: dict[str, pd.DataFrame]) -> go.Figure | None:
    """Build revenue, net income, and free cash flow bar chart."""

    income = statements.get("income_statement", pd.DataFrame())
    cash_flow = statements.get("cash_flow", pd.DataFrame())
    rows = []
    for label, statement, candidates in [
        ("Revenue", income, ("Total Revenue", "Operating Revenue")),
        ("Net Income", income, ("Net Income", "Net Income Common Stockholders")),
        ("Free Cash Flow", cash_flow, ("Free Cash Flow",)),
    ]:
        series = statement_row(statement, candidates)
        if series is None:
            continue
        for date, value in series.dropna().sort_index().items():
            rows.append({"date": date, "metric": label, "value": value})

    if not rows:
        return None

    frame = pd.DataFrame(rows)
    figure = go.Figure()
    for metric, group in frame.groupby("metric"):
        figure.add_trace(go.Bar(x=group["date"], y=group["value"], name=metric))
    figure.update_layout(
        barmode="group",
        height=420,
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        xaxis_title="Fiscal Year",
        yaxis_title="Value",
        legend={"orientation": "h"},
    )
    return figure


def statement_row(statement: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    """Find a statement row by possible labels."""

    if statement.empty:
        return None
    index_map = {str(index).lower(): index for index in statement.index}
    for candidate in candidates:
        key = candidate.lower()
        if key in index_map:
            return pd.to_numeric(statement.loc[index_map[key]], errors="coerce")
    return None


def build_valuation_table(
    method: str,
    latest_close: float,
    market_cap: float | None,
    base_eps: float,
    base_pe: float,
    base_revenue_billions: float | None,
    base_ps: float | None,
) -> pd.DataFrame:
    """Build selected valuation scenario table."""

    shares_outstanding = None
    if market_cap and latest_close > 0:
        shares_outstanding = market_cap / latest_close

    assumptions = build_scenarios(
        base_forward_eps=base_eps,
        base_pe_multiple=base_pe,
        base_forward_revenue=None if base_revenue_billions is None else base_revenue_billions * 1_000_000_000,
        base_ps_multiple=base_ps,
        shares_outstanding=shares_outstanding,
    )
    if method == "PE":
        return estimate_by_pe(assumptions)
    if method == "PS":
        return estimate_by_ps(assumptions)
    return blended_valuation(assumptions)


def build_peer_table(peers: list[str], period: str, interval: str) -> pd.DataFrame:
    """Build a peer comparison table from technical and financial metrics."""

    rows = []
    for peer in peers:
        data, error = load_price_data(peer, period, interval)
        profile, _ = load_profile(peer)
        _, financial_metrics, _ = load_financial_data(peer)
        if data is None:
            rows.append(
                {
                    "Ticker": peer,
                    "Market Cap": format_value(profile.get("marketCap"), "marketCap"),
                    "Latest Close": "N/A",
                    "RSI": "N/A",
                    "Revenue Growth": format_value(
                        financial_metrics.get("revenue_growth_yoy"), "revenue_growth_yoy"
                    ),
                    "Net Margin": format_value(financial_metrics.get("net_margin"), "net_margin"),
                    "Trend": error or "N/A",
                }
            )
            continue
        metrics = calculate_technical_metrics(data)
        rows.append(
            {
                "Ticker": peer,
                "Market Cap": format_value(profile.get("marketCap"), "marketCap"),
                "Latest Close": format_value(metrics["latest_close"], "latest_close"),
                "RSI": format_value(metrics["RSI"], "RSI"),
                "Revenue Growth": format_value(
                    financial_metrics.get("revenue_growth_yoy"), "revenue_growth_yoy"
                ),
                "Net Margin": format_value(financial_metrics.get("net_margin"), "net_margin"),
                "Trend": metrics["trend"].replace("_", " "),
            }
        )
    return pd.DataFrame(rows)


def render_metric_cards(technical_metrics: dict[str, Any], financial_metrics: dict[str, float | None]) -> None:
    """Render top-level Streamlit metric cards."""

    columns = st.columns(6)
    cards = [
        ("Close", format_value(technical_metrics["latest_close"], "latest_close")),
        ("RSI", format_value(technical_metrics["RSI"], "RSI")),
        ("Volatility", format_value(technical_metrics["annualized_volatility"], "annualized_volatility")),
        ("Revenue Growth", format_value(financial_metrics.get("revenue_growth_yoy"), "revenue_growth_yoy")),
        ("Net Margin", format_value(financial_metrics.get("net_margin"), "net_margin")),
        ("FCF Margin", format_value(financial_metrics.get("fcf_margin"), "fcf_margin")),
    ]
    for column, (label, value) in zip(columns, cards, strict=True):
        column.metric(label, value)


def format_optional_percent(value: float | None) -> str:
    """Format optional percentage values for UI display."""

    return "N/A" if value is None or pd.isna(value) else f"{value:.1%}"


def metrics_to_display_frame(metrics: dict[str, float | None]) -> pd.DataFrame:
    """Convert metric dictionaries into readable dashboard rows."""

    rows = [
        {"Metric": humanize_label(key), "Value": format_value(value, key)}
        for key, value in metrics.items()
    ]
    return pd.DataFrame(rows)


def valuation_to_display_frame(valuation_table: pd.DataFrame) -> pd.DataFrame:
    """Convert valuation output into a compact dashboard table."""

    display = valuation_table.copy()
    if "target_price" in display.columns:
        display["target_price"] = display["target_price"].map(lambda value: format_value(value, "target_price"))
        display = display.rename(columns={"target_price": "Target Price"})
    if "assumed_inputs" in display.columns:
        display = display.drop(columns=["assumed_inputs"])
    display = display.rename(columns={column: humanize_label(column) for column in display.columns})
    return display


def main() -> None:
    """Render the Streamlit dashboard."""

    settings = load_yaml(CONFIG_DIR / "settings.yaml")
    peers_config = load_yaml(CONFIG_DIR / "peers.yaml")

    st.set_page_config(page_title="Stock Research Copilot", layout="wide")

    with st.sidebar:
        default_ticker = str(settings.get("default_ticker", "AAPL"))
        ticker = st.text_input("Ticker", value=default_ticker).strip().upper() or default_ticker
        selected_period_label = st.selectbox("Period", list(PERIOD_OPTIONS), index=2)
        period = PERIOD_OPTIONS[selected_period_label]
        interval = str(settings.get("default_interval", "1d"))
        default_peers = ", ".join(peers_config.get(ticker, []))
        peer_text = st.text_input("Peers", value=default_peers)
        valuation_method = st.selectbox("Valuation", ["blended", "PE", "PS"])
        with st.expander("Scenario Assumptions"):
            base_eps = st.number_input("Forward EPS", min_value=0.0, value=10.0, step=0.1)
            base_pe = st.number_input("Base PE", min_value=0.0, value=22.0, step=0.5)
            base_revenue_billions = st.number_input(
                "Forward Revenue ($B)",
                min_value=0.0,
                value=400.0,
                step=5.0,
            )
            base_ps = st.number_input("Base P/S", min_value=0.0, value=6.0, step=0.1)

    price_data, price_warning = load_price_data(ticker, period, interval)
    profile, profile_warning = load_profile(ticker)

    st.title(f"{profile.get('name') or ticker} ({ticker})")
    st.caption(
        " | ".join(
            item
            for item in [
                str(profile.get("sector") or ""),
                str(profile.get("industry") or ""),
                f"Market Cap: {format_value(profile.get('marketCap'), 'marketCap')}",
                str(profile.get("currency") or ""),
            ]
            if item
        )
    )

    if price_warning:
        st.warning(price_warning)
    if profile_warning:
        st.warning(profile_warning)

    if price_data is None:
        st.error("Price data is unavailable for this ticker.")
        st.caption(DISCLAIMER)
        return

    statements, financial_metrics, financial_summary = load_financial_data(ticker)
    technical_metrics = calculate_technical_metrics(price_data)
    valuation_table = build_valuation_table(
        method=valuation_method,
        latest_close=technical_metrics["latest_close"],
        market_cap=profile.get("marketCap"),
        base_eps=base_eps,
        base_pe=base_pe,
        base_revenue_billions=base_revenue_billions if base_revenue_billions > 0 else None,
        base_ps=base_ps if base_ps > 0 else None,
    )
    valuation_summary = summarize_valuation(
        valuation_table,
        current_price=technical_metrics["latest_close"],
    )

    render_metric_cards(technical_metrics, financial_metrics)

    st.subheader("Price Trend")
    st.plotly_chart(build_price_chart(price_data), width="stretch")

    st.subheader("Financial Performance")
    st.write(financial_summary)
    financial_chart = build_financial_charts(statements)
    if financial_chart is not None:
        st.plotly_chart(financial_chart, width="stretch")
    if financial_metrics:
        st.dataframe(metrics_to_display_frame(financial_metrics), width="stretch", hide_index=True)

    st.subheader("Valuation Scenarios")
    st.write(valuation_summary)
    st.dataframe(valuation_to_display_frame(valuation_table), width="stretch", hide_index=True)

    peers = parse_peer_input(peer_text)
    st.subheader("Peer Comparison")
    peer_frame = pd.DataFrame()
    if peers:
        peer_frame = build_peer_table(peers, period, interval)
        st.dataframe(peer_frame, width="stretch", hide_index=True)
    else:
        st.write("No configured peers.")

    report_context = build_report_context(
        ticker=ticker,
        price_history=price_data,
        company_profile=profile,
        financial_metrics=financial_metrics,
        financial_summary=financial_summary,
        valuation_table=valuation_table,
        valuation_summary=valuation_summary,
        peer_comparison=format_dataframe_markdown(peer_frame) if not peer_frame.empty else None,
    )
    markdown_report = render_markdown_report(report_context)

    st.subheader("Research Report")
    st.markdown(markdown_report)
    st.download_button(
        "Download Markdown",
        data=markdown_report,
        file_name=f"{ticker}_report.md",
        mime="text/markdown",
    )
    if st.button("Save Report"):
        output_path = save_report(markdown_report, ticker, REPORT_DIR)
        st.success(f"Saved to {output_path}")

    st.caption(DISCLAIMER)
    st.caption("Data sources: Yahoo Finance via yfinance. Financial data may be incomplete or unavailable.")


if __name__ == "__main__":
    main()
