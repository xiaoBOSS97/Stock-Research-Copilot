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
from src.analysis.market_implied_expectations import (
    scenario_price_deviation,
    solve_implied_fcf_growth,
    summarize_market_implied_expectations,
)
from src.analysis.technical_analysis import add_technical_indicators, calculate_technical_metrics
from src.analysis.valuation import (
    blended_valuation,
    blended_valuation_with_dcf,
    build_dcf_scenarios,
    build_scenarios,
    estimate_by_dcf,
    estimate_by_pe,
    estimate_by_ps,
    summarize_valuation,
)
from src.data_loader.financial_loader import FinancialDataError, get_financial_statements
from src.data_loader.price_loader import PriceDataError, get_company_profile, get_price_history
from src.data_loader.sec_loader import SecFilingError, download_filing, get_latest_filing
from src.data_loader.transcript_loader import (
    TranscriptDownloadError,
    download_and_save_alpha_vantage_transcript,
    normalize_quarter,
)
from src.preprocessing.filing_parser import parse_filing_file, save_parsed_filing
from src.report.report_generator import (
    DISCLAIMER,
    build_report_context,
    format_dataframe_markdown,
    latest_sec_filing_summary,
    render_markdown_report,
    save_report,
)
from src.rag.filing_qa import answer_filing_question, list_parsed_filing_files, load_parsed_filing_sections
from src.rag.transcript_analysis import (
    analyze_transcript,
    list_transcript_files,
    load_transcript_text,
    save_transcript_text,
    search_transcript,
)
from src.utils.formatting import calculate_upside_downside, format_value, humanize_label
from src.utils.glossary import term_help


CONFIG_DIR = PROJECT_ROOT / "config"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
FILING_DIR = PROJECT_ROOT / "data" / "filings"
PARSED_FILING_DIR = PROJECT_ROOT / "data" / "processed" / "filings"
PERIOD_OPTIONS = {"1Y": "1y", "3Y": "3y", "5Y": "5y", "Max": "max"}
EXAMPLE_TICKERS = ("AAPL", "NVDA", "MSFT", "TSLA")
REPORT_SCHEMA_VERSION = "market-implied-v1"
FALLBACK_VALUATION_DEFAULTS = {
    "base_eps": 10.0,
    "base_pe": 22.0,
    "base_revenue_billions": 400.0,
    "base_ps": 6.0,
    "base_free_cash_flow_billions": 100.0,
    "net_debt_billions": 0.0,
}
PERFORMANCE_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Growth", ("revenue_growth_yoy", "net_income_growth_yoy", "revenue_cagr")),
    ("Profitability", ("gross_margin", "operating_margin", "net_margin", "fcf_margin")),
    ("Returns", ("roe", "roa", "roic")),
    ("Balance Sheet", ("debt_to_equity", "current_ratio", "interest_coverage", "free_cash_flow")),
)


def apply_dashboard_style() -> None:
    """Apply small dashboard polish without fighting Streamlit's native controls."""

    st.markdown(
        """
        <style>
        .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
        h1 { margin-bottom: 0.25rem; }
        h2, h3 { margin-top: 1.6rem; }
        div[data-testid="stMetric"] { padding: 0.25rem 0; }
        div[data-testid="stMetricLabel"] > div { font-weight: 650; }
        .src-caption {
            color: #9ca3af;
            font-size: 0.92rem;
            margin: -0.25rem 0 0.75rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


@st.cache_data(show_spinner=False)
def load_saved_transcript(path_text: str) -> str:
    """Load a saved transcript by path for Streamlit caching."""

    return load_transcript_text(path_text)


@st.cache_data(show_spinner=False)
def load_saved_filing_sections(path_text: str):
    """Load parsed filing sections by path for Streamlit caching."""

    return load_parsed_filing_sections(path_text)


def parse_peer_input(raw_text: str) -> list[str]:
    """Parse comma-separated peer ticker input."""

    return [item.strip().upper() for item in raw_text.split(",") if item.strip()]


def default_period_index(settings: dict[str, Any]) -> int:
    """Return the configured default period index for the sidebar selector."""

    default_period = str(settings.get("default_period", "5y"))
    options = list(PERIOD_OPTIONS.values())
    return options.index(default_period) if default_period in options else 2


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


def latest_statement_value(statement: pd.DataFrame, candidates: tuple[str, ...]) -> float | None:
    """Return the newest numeric value for the first matching statement row."""

    row = statement_row(statement, candidates)
    if row is None:
        return None
    values = row.dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def first_numeric(*values: Any, positive: bool = False) -> float | None:
    """Return the first finite numeric value, optionally requiring it to be positive."""

    for value in values:
        if value is None:
            continue
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            continue
        result = float(numeric)
        if positive and result <= 0:
            continue
        return result
    return None


def to_billions(value: float | None) -> float | None:
    """Convert an absolute value to billions for sidebar inputs."""

    return None if value is None else value / 1_000_000_000


def valuation_input_defaults(
    profile: dict[str, Any],
    statements: dict[str, pd.DataFrame],
    financial_metrics: dict[str, float | None],
) -> dict[str, float | None]:
    """Build valuation defaults from live/TTM fields with conservative fallbacks."""

    income = statements.get("income_statement", pd.DataFrame())
    total_revenue = first_numeric(
        profile.get("totalRevenue"),
        latest_statement_value(income, ("Total Revenue", "Operating Revenue")),
        positive=True,
    )
    market_cap = first_numeric(profile.get("marketCap"), positive=True)
    total_debt = first_numeric(profile.get("totalDebt"))
    total_cash = first_numeric(profile.get("totalCash"))

    defaults = {
        "base_eps": first_numeric(
            profile.get("forwardEps"),
            profile.get("trailingEps"),
            positive=True,
        ),
        "base_pe": first_numeric(
            profile.get("forwardPE"),
            profile.get("trailingPE"),
            positive=True,
        ),
        "base_revenue_billions": to_billions(total_revenue),
        "base_ps": first_numeric(profile.get("priceToSalesTrailing12Months"), positive=True),
        "base_free_cash_flow_billions": to_billions(
            first_numeric(
                profile.get("freeCashflow"),
                financial_metrics.get("free_cash_flow"),
                positive=True,
            )
        ),
        "shares_outstanding": first_numeric(profile.get("sharesOutstanding"), positive=True),
        "net_debt_billions": (
            to_billions(total_debt - (total_cash or 0.0)) if total_debt is not None else None
        ),
    }
    if defaults["base_ps"] is None and market_cap is not None and total_revenue:
        defaults["base_ps"] = market_cap / total_revenue

    return {
        key: defaults[key] if defaults.get(key) is not None else fallback
        for key, fallback in FALLBACK_VALUATION_DEFAULTS.items()
    } | {"shares_outstanding": defaults["shares_outstanding"]}


def build_valuation_table(
    method: str,
    latest_close: float,
    market_cap: float | None,
    shares_outstanding: float | None,
    base_eps: float,
    base_pe: float,
    base_revenue_billions: float | None,
    base_ps: float | None,
    base_free_cash_flow_billions: float | None = None,
    dcf_growth_rate: float = 0.05,
    dcf_discount_rate: float = 0.10,
    dcf_terminal_growth_rate: float = 0.025,
    net_debt_billions: float = 0.0,
) -> pd.DataFrame:
    """Build selected valuation scenario table."""

    if shares_outstanding is None and market_cap and latest_close > 0:
        shares_outstanding = market_cap / latest_close

    assumptions = build_scenarios(
        base_forward_eps=base_eps,
        base_pe_multiple=base_pe,
        base_forward_revenue=None if base_revenue_billions is None else base_revenue_billions * 1_000_000_000,
        base_ps_multiple=base_ps,
        shares_outstanding=shares_outstanding,
    )
    dcf_assumptions = build_dcf_scenarios(
        base_free_cash_flow=None
        if base_free_cash_flow_billions is None
        else base_free_cash_flow_billions * 1_000_000_000,
        base_growth_rate=dcf_growth_rate,
        discount_rate=dcf_discount_rate,
        terminal_growth_rate=dcf_terminal_growth_rate,
        net_debt=net_debt_billions * 1_000_000_000,
        shares_outstanding=shares_outstanding,
    )
    if method == "PE":
        return estimate_by_pe(assumptions)
    if method == "PS":
        return estimate_by_ps(assumptions)
    if method == "DCF":
        return estimate_by_dcf(dcf_assumptions)
    if method == "blended + DCF":
        return blended_valuation_with_dcf(assumptions, dcf_assumptions=dcf_assumptions)
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
    for column, (label, value, help_text) in zip(
        columns,
        metric_card_specs(technical_metrics, financial_metrics),
        strict=True,
    ):
        column.metric(label, value, help=help_text)


def metric_card_specs(
    technical_metrics: dict[str, Any],
    financial_metrics: dict[str, float | None],
) -> list[tuple[str, str, str]]:
    """Return labels, values, and native Streamlit help text for metric cards."""

    cards = [
        ("Close", format_value(technical_metrics["latest_close"], "latest_close")),
        ("RSI", format_value(technical_metrics["RSI"], "RSI")),
        ("Volatility", format_value(technical_metrics["annualized_volatility"], "annualized_volatility")),
        ("Revenue Growth", format_value(financial_metrics.get("revenue_growth_yoy"), "revenue_growth_yoy")),
        ("Net Margin", format_value(financial_metrics.get("net_margin"), "net_margin")),
        ("FCF Margin", format_value(financial_metrics.get("fcf_margin"), "fcf_margin")),
    ]
    return [(label, value, term_help(label)) for label, value in cards]


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


def performance_metric_specs(metrics: dict[str, float | None]) -> list[tuple[str, str, str]]:
    """Return native Streamlit metric specs for the Financial Performance section."""

    return [
        (humanize_label(key), format_value(value, key), term_help(humanize_label(key)))
        for key, value in metrics.items()
    ]


def grouped_performance_metric_specs(
    metrics: dict[str, float | None],
) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Group financial metrics into dashboard-friendly sections."""

    grouped = []
    used_keys = set()
    for group_name, keys in PERFORMANCE_METRIC_GROUPS:
        specs = [
            (humanize_label(key), format_value(metrics.get(key), key), term_help(humanize_label(key)))
            for key in keys
            if key in metrics
        ]
        if specs:
            grouped.append((group_name, specs))
            used_keys.update(keys)

    remaining = [
        (humanize_label(key), format_value(value, key), term_help(humanize_label(key)))
        for key, value in metrics.items()
        if key not in used_keys
    ]
    if remaining:
        grouped.append(("Other", remaining))

    return grouped


def render_performance_metrics(metrics: dict[str, float | None]) -> None:
    """Render financial performance metrics with native Streamlit help icons."""

    groups = grouped_performance_metric_specs(metrics)
    if not groups:
        st.info("Financial performance metrics are unavailable for this ticker.")
        return

    tabs = st.tabs([group_name for group_name, _ in groups])
    for tab, (_, specs) in zip(tabs, groups, strict=True):
        with tab:
            for index in range(0, len(specs), 4):
                columns = st.columns(4)
                for column, (label, value, help_text) in zip(columns, specs[index : index + 4], strict=False):
                    column.metric(label, value, help=help_text)


def render_transcript_analysis(ticker: str) -> None:
    """Render local earnings call transcript upload, analysis, and retrieval."""

    st.markdown("#### Earnings Call Transcript")
    st.caption("Use management commentary to check demand, margin, guidance, and risk language.")
    quarter = st.text_input(
        "Fiscal quarter",
        value="2024Q1",
        help="Quarter for automatic Alpha Vantage transcript download. Format: YYYYQ#, for example 2024Q1.",
    )
    if st.button("Download Transcript"):
        try:
            saved_path = download_and_save_alpha_vantage_transcript(
                ticker,
                normalize_quarter(quarter),
                output_dir=TRANSCRIPT_DIR,
            )
            st.success(f"Downloaded transcript to {saved_path}")
            load_saved_transcript.clear()
        except (TranscriptDownloadError, ValueError) as exc:
            st.warning(f"Automatic transcript download unavailable: {exc}")

    uploaded_file = st.file_uploader(
        "Upload transcript",
        type=["txt", "md"],
        help="Upload a plain-text or Markdown earnings call transcript for local analysis.",
    )
    if uploaded_file is not None:
        transcript_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
        saved_path = save_transcript_text(ticker, uploaded_file.name, transcript_text, TRANSCRIPT_DIR)
        st.success(f"Saved transcript to {saved_path}")

    transcript_files = list_transcript_files(ticker, TRANSCRIPT_DIR)
    if not transcript_files:
        st.info("No transcript uploaded yet. Add a .txt or .md earnings call transcript to analyze management commentary.")
        return

    selected_path = st.selectbox(
        "Transcript",
        transcript_files,
        format_func=lambda path: path.name,
        help="Saved local transcripts for the current ticker.",
    )
    transcript_text = load_saved_transcript(str(selected_path))
    analysis = analyze_transcript(transcript_text)

    col1, col2, col3 = st.columns(3)
    col1.metric("Words", f"{analysis['word_count']:,}")
    col2.metric("Chunks", str(analysis["chunk_count"]))
    col3.metric("Tone", analysis["management_tone"])

    st.markdown("##### Topic Signals")
    st.dataframe(analysis["topic_table"], width="stretch", hide_index=True)

    suggested_questions = analysis["key_questions"]
    default_question = suggested_questions[0] if suggested_questions else "What did management emphasize?"
    question = st.text_input(
        "Ask the transcript",
        value=default_question,
        help="Keyword retrieval over the uploaded transcript. Results are source passages, not financial advice.",
    )
    if question:
        results = search_transcript(transcript_text, question)
        if results.empty:
            st.info("No matching transcript passages found.")
        else:
            st.dataframe(results, width="stretch", hide_index=True)


def render_filing_qa(ticker: str) -> None:
    """Render SEC filing download, parse, and source-grounded Q&A."""

    st.markdown("#### SEC Filing Q&A")
    st.caption("Download a public SEC filing, parse the core sections, then ask questions against cited passages.")
    selected_form = st.selectbox(
        "Filing type",
        ["10-K", "10-Q", "8-K"],
        help="SEC filing type to download and parse. 10-K is annual, 10-Q is quarterly, 8-K is a major event filing.",
    )
    if st.button("Download and Parse Latest Filing"):
        try:
            filing = get_latest_filing(ticker, form=selected_form)
            filing_path = download_filing(filing, output_dir=FILING_DIR)
            sections = parse_filing_file(filing_path)
            parsed_path = save_parsed_filing(
                sections,
                ticker=ticker,
                accession_number=filing.accession_number,
                output_dir=PARSED_FILING_DIR,
            )
            load_saved_filing_sections.clear()
            st.success(f"Parsed {selected_form} filing to {parsed_path}")
        except (SecFilingError, ValueError) as exc:
            st.warning(f"SEC filing download or parsing unavailable: {exc}")

    parsed_files = list_parsed_filing_files(ticker, PARSED_FILING_DIR)
    if not parsed_files:
        st.info("No parsed SEC filing yet. Download and parse a filing to ask source-grounded questions.")
        return

    selected_path = st.selectbox(
        "Parsed filing",
        parsed_files,
        format_func=lambda path: path.name,
        help="Parsed local SEC filing sections for the current ticker.",
    )
    sections = load_saved_filing_sections(str(selected_path))
    st.caption(f"{len(sections)} parsed sections available.")

    question = st.text_input(
        "Ask the filing",
        value="What are the main risk factors?",
        help="Retrieves passages from the parsed SEC filing and answers only from those sources.",
    )
    if not question:
        return

    result = answer_filing_question(sections, question, ticker=ticker, filing_type=selected_form)
    if result.refused:
        st.info(result.answer)
        return

    st.markdown(result.answer)
    st.dataframe(result.sources_frame(), width="stretch", hide_index=True)


def render_source_research(ticker: str) -> None:
    """Group source-backed qualitative research tools."""

    st.subheader("Source Research")
    st.markdown(
        '<div class="src-caption">Qualitative evidence from transcripts and SEC filings. '
        "These sections are retrieval aids, not standalone conclusions.</div>",
        unsafe_allow_html=True,
    )
    transcript_tab, filing_tab = st.tabs(["Earnings Call", "SEC Filing"])
    with transcript_tab:
        render_transcript_analysis(ticker)
    with filing_tab:
        render_filing_qa(ticker)


def is_current_report_preview(markdown_report: str | None, ticker: str, report_ticker: str | None, schema_version: str | None) -> bool:
    """Return whether the cached report preview matches the current report shape."""

    return (
        bool(markdown_report)
        and report_ticker == ticker
        and schema_version == REPORT_SCHEMA_VERSION
        and "## Executive Summary" in markdown_report
    )


def valuation_to_display_frame(valuation_table: pd.DataFrame, current_price: float | None = None) -> pd.DataFrame:
    """Convert valuation output into a compact dashboard table."""

    display = valuation_table.copy()
    if "target_price" in display.columns:
        display["upside_downside"] = display["target_price"].map(
            lambda target_price: calculate_upside_downside(target_price, current_price)
        )
    if "target_price" in display.columns:
        display["target_price"] = display["target_price"].map(lambda value: format_value(value, "target_price"))
        display = display.rename(columns={"target_price": "Target Price"})
    if "upside_downside" in display.columns:
        display["upside_downside"] = display["upside_downside"].map(
            lambda value: format_value(value, "upside_downside")
        )
    if "assumed_inputs" in display.columns:
        display = display.drop(columns=["assumed_inputs"])
    display = display.rename(columns={column: humanize_label(column) for column in display.columns})
    return display


def market_implied_to_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Format market-implied scenario deviation rows for the dashboard."""

    display = frame.copy()
    for column in ("target_price", "price_gap"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: format_value(value, column))
    if "upside_downside" in display.columns:
        display["upside_downside"] = display["upside_downside"].map(
            lambda value: format_value(value, "upside_downside")
        )
    return display.rename(columns={column: humanize_label(column) for column in display.columns})


def market_implied_status_caption(status: str) -> str:
    """Return short UI copy for the implied growth solver status."""

    if status == "ok":
        return "Solved within the configured DCF growth range."
    if status == "out_of_bounds":
        return "Current price sits outside the configured growth bounds."
    return "Solver needs positive current price, FCF, discount rate, terminal growth, and share count."


def main() -> None:
    """Render the Streamlit dashboard."""

    settings = load_yaml(CONFIG_DIR / "settings.yaml")
    peers_config = load_yaml(CONFIG_DIR / "peers.yaml")

    st.set_page_config(page_title="Stock Research Copilot", layout="wide")
    apply_dashboard_style()
    st.session_state.setdefault("markdown_report", None)
    st.session_state.setdefault("markdown_report_ticker", None)
    st.session_state.setdefault("markdown_report_schema_version", None)

    with st.sidebar:
        default_ticker = str(settings.get("default_ticker", "AAPL"))
        ticker = (
            st.text_input("Ticker", value=default_ticker, help=term_help("Ticker")).strip().upper()
            or default_ticker
        )
        st.caption("Try: " + ", ".join(EXAMPLE_TICKERS))
        selected_period_label = st.selectbox(
            "Period",
            list(PERIOD_OPTIONS),
            index=default_period_index(settings),
            help="Historical price range used for price charts and technical indicators.",
        )
        period = PERIOD_OPTIONS[selected_period_label]
        interval = str(settings.get("default_interval", "1d"))
        default_peers = ", ".join(peers_config.get(ticker, []))
        peer_text = st.text_input("Peers", value=default_peers, help="Comparable tickers for peer metrics.")
        valuation_method = st.selectbox(
            "Valuation Method",
            ["blended", "blended + DCF", "PE", "PS", "DCF"],
            help="Scenario valuation method based on explicit assumptions.",
        )

        with st.spinner(f"Loading {ticker} market data..."):
            price_data, price_warning = load_price_data(ticker, period, interval)
            profile, profile_warning = load_profile(ticker)
        with st.spinner(f"Loading {ticker} financial statements..."):
            statements, financial_metrics, financial_summary = load_financial_data(ticker)

        valuation_defaults = valuation_input_defaults(profile, statements, financial_metrics)
        with st.expander("Scenario Assumptions"):
            st.caption("Defaults use live/TTM data when available; missing fields fall back to MVP assumptions.")
            base_eps = st.number_input(
                "Forward EPS",
                min_value=0.0,
                value=float(valuation_defaults["base_eps"] or FALLBACK_VALUATION_DEFAULTS["base_eps"]),
                step=0.1,
                help=term_help("Forward EPS"),
                key=f"{ticker}_base_eps",
            )
            base_pe = st.number_input(
                "Base P/E",
                min_value=0.0,
                value=float(valuation_defaults["base_pe"] or FALLBACK_VALUATION_DEFAULTS["base_pe"]),
                step=0.5,
                help=term_help("P/E"),
                key=f"{ticker}_base_pe",
            )
            base_revenue_billions = st.number_input(
                "Forward Revenue ($B)",
                min_value=0.0,
                value=float(
                    valuation_defaults["base_revenue_billions"]
                    or FALLBACK_VALUATION_DEFAULTS["base_revenue_billions"]
                ),
                step=5.0,
                help=term_help("Forward Revenue"),
                key=f"{ticker}_base_revenue_billions",
            )
            base_ps = st.number_input(
                "Base P/S",
                min_value=0.0,
                value=float(valuation_defaults["base_ps"] or FALLBACK_VALUATION_DEFAULTS["base_ps"]),
                step=0.1,
                help=term_help("P/S"),
                key=f"{ticker}_base_ps",
            )
            base_fcf_billions = st.number_input(
                "Base Free Cash Flow ($B)",
                min_value=0.0,
                value=float(
                    valuation_defaults["base_free_cash_flow_billions"]
                    or FALLBACK_VALUATION_DEFAULTS["base_free_cash_flow_billions"]
                ),
                step=5.0,
                help=term_help("Base Free Cash Flow"),
                key=f"{ticker}_base_fcf_billions",
            )
            dcf_growth_rate = st.number_input(
                "DCF Growth Rate",
                min_value=-0.50,
                max_value=0.50,
                value=0.05,
                step=0.005,
                help=term_help("Growth Rate"),
                key=f"{ticker}_dcf_growth_rate",
            )
            dcf_discount_rate = st.number_input(
                "DCF Discount Rate",
                min_value=0.001,
                max_value=0.50,
                value=0.10,
                step=0.005,
                help=term_help("Discount Rate"),
                key=f"{ticker}_dcf_discount_rate",
            )
            dcf_terminal_growth_rate = st.number_input(
                "DCF Terminal Growth",
                min_value=0.0,
                max_value=0.10,
                value=0.025,
                step=0.005,
                help=term_help("Terminal Growth Rate"),
                key=f"{ticker}_dcf_terminal_growth_rate",
            )
            net_debt_billions = st.number_input(
                "Net Debt ($B)",
                value=float(
                    valuation_defaults["net_debt_billions"]
                    or FALLBACK_VALUATION_DEFAULTS["net_debt_billions"]
                ),
                step=5.0,
                help=term_help("Net Debt"),
                key=f"{ticker}_net_debt_billions",
            )

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

    technical_metrics = calculate_technical_metrics(price_data)
    valuation_table = build_valuation_table(
        method=valuation_method,
        latest_close=technical_metrics["latest_close"],
        market_cap=profile.get("marketCap"),
        shares_outstanding=valuation_defaults.get("shares_outstanding"),
        base_eps=base_eps,
        base_pe=base_pe,
        base_revenue_billions=base_revenue_billions if base_revenue_billions > 0 else None,
        base_ps=base_ps if base_ps > 0 else None,
        base_free_cash_flow_billions=base_fcf_billions if base_fcf_billions > 0 else None,
        dcf_growth_rate=dcf_growth_rate,
        dcf_discount_rate=dcf_discount_rate,
        dcf_terminal_growth_rate=dcf_terminal_growth_rate,
        net_debt_billions=net_debt_billions,
    )
    valuation_summary = summarize_valuation(
        valuation_table,
        current_price=technical_metrics["latest_close"],
    )
    implied_growth = solve_implied_fcf_growth(
        current_price=technical_metrics["latest_close"],
        base_free_cash_flow=base_fcf_billions * 1_000_000_000 if base_fcf_billions > 0 else None,
        discount_rate=dcf_discount_rate,
        terminal_growth_rate=dcf_terminal_growth_rate,
        shares_outstanding=valuation_defaults.get("shares_outstanding")
        or (
            profile.get("marketCap") / technical_metrics["latest_close"]
            if profile.get("marketCap") and technical_metrics["latest_close"] > 0
            else None
        ),
        net_debt=net_debt_billions * 1_000_000_000,
    )
    scenario_deviation = scenario_price_deviation(valuation_table, technical_metrics["latest_close"])
    market_implied_summary = summarize_market_implied_expectations(implied_growth, scenario_deviation)

    render_metric_cards(technical_metrics, financial_metrics)
    st.divider()

    st.subheader("Price Trend")
    st.caption("Price history with moving averages for trend context.")
    st.plotly_chart(build_price_chart(price_data), width="stretch")

    st.subheader("Financial Performance")
    st.caption("Core growth, profitability, return, leverage, and cash flow metrics from available statements.")
    st.write(financial_summary)
    financial_chart = build_financial_charts(statements)
    if financial_chart is not None:
        st.plotly_chart(financial_chart, width="stretch")
    if financial_metrics:
        render_performance_metrics(financial_metrics)

    st.divider()
    render_source_research(ticker)

    st.divider()
    st.subheader("Valuation Scenarios")
    st.caption("Bear/Base/Bull estimates from explicit assumptions. These are scenario outputs, not predictions.")
    st.write(valuation_summary)
    st.dataframe(
        valuation_to_display_frame(valuation_table, current_price=technical_metrics["latest_close"]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Market-Implied Expectations")
    st.caption("Reverse-solves what the current price implies under the configured DCF assumptions.")
    st.write(market_implied_summary)
    cols = st.columns(3)
    cols[0].metric(
        "Implied FCF Growth",
        format_value(implied_growth.implied_growth_rate, "implied_growth_rate"),
        help=term_help("Implied FCF Growth"),
    )
    cols[1].metric(
        "Model Price",
        format_value(implied_growth.model_price, "target_price"),
        help="DCF price produced by the reverse-solved implied growth rate.",
    )
    cols[2].metric(
        "Status",
        implied_growth.status.replace("_", " "),
        help=market_implied_status_caption(implied_growth.status),
    )
    st.dataframe(market_implied_to_display_frame(scenario_deviation), width="stretch", hide_index=True)

    st.divider()
    peers = parse_peer_input(peer_text)
    st.subheader("Peer Comparison")
    st.caption("Quick comparison against selected tickers using the same public-data pipeline.")
    peer_frame = pd.DataFrame()
    if peers:
        peer_frame = build_peer_table(peers, period, interval)
        st.dataframe(peer_frame, width="stretch", hide_index=True)
    else:
        st.write("No configured peers.")

    st.subheader("Research Report")
    st.caption("Generate a Markdown preview after reviewing assumptions, peers, and source-backed sections.")
    if st.button("Generate Report Preview", type="primary"):
        with st.spinner("Generating report preview..."):
            report_context = build_report_context(
                ticker=ticker,
                price_history=price_data,
                company_profile=profile,
                financial_metrics=financial_metrics,
                financial_summary=financial_summary,
                valuation_table=valuation_table,
                valuation_summary=valuation_summary,
                market_implied_summary=market_implied_summary,
                market_implied_table=scenario_deviation,
                implied_growth=implied_growth.to_dict(),
                peer_comparison=format_dataframe_markdown(peer_frame) if not peer_frame.empty else None,
                sec_filing_summary=latest_sec_filing_summary(ticker, PARSED_FILING_DIR),
            )
            st.session_state["markdown_report"] = render_markdown_report(report_context)
            st.session_state["markdown_report_ticker"] = ticker
            st.session_state["markdown_report_schema_version"] = REPORT_SCHEMA_VERSION

    markdown_report = st.session_state.get("markdown_report")
    report_ticker = st.session_state.get("markdown_report_ticker")
    report_schema_version = st.session_state.get("markdown_report_schema_version")
    if is_current_report_preview(markdown_report, ticker, report_ticker, report_schema_version):
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
    elif markdown_report and report_ticker == ticker:
        st.info("This report preview is from an older format. Generate a fresh preview to see the latest sections.")
    elif markdown_report:
        st.info("Generate a fresh report preview for the current ticker.")
    else:
        st.info("No report preview generated yet.")

    st.caption(DISCLAIMER)
    st.caption("Data sources: Yahoo Finance via yfinance. Financial data may be incomplete or unavailable.")


if __name__ == "__main__":
    main()
