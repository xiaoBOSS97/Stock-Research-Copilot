from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from src.analysis.valuation import blended_valuation, build_scenarios
from src.data_loader.financial_loader import FinancialDataError
from src.report import report_generator


def sample_price_history() -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100 + index for index in range(220)]},
        index=pd.date_range("2025-01-01", periods=220, freq="D"),
    )


def sample_profile() -> dict[str, object]:
    return {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 123,
        "currency": "USD",
    }


def sample_valuation_table() -> pd.DataFrame:
    assumptions = build_scenarios(
        base_forward_eps=10.0,
        base_pe_multiple=20.0,
        base_forward_revenue=100_000.0,
        base_ps_multiple=5.0,
        shares_outstanding=1_000.0,
    )
    return blended_valuation(assumptions)


def test_build_report_context_contains_required_sections() -> None:
    context = report_generator.build_report_context(
        ticker="AAPL",
        price_history=sample_price_history(),
        company_profile=sample_profile(),
        financial_metrics={"revenue_growth_yoy": 0.2, "net_margin": 0.25},
        valuation_table=sample_valuation_table(),
        peer_comparison="| Ticker | Latest Close |\n| --- | --- |\n| MSFT | 100.00 |",
        generated_at=datetime(2026, 5, 6, tzinfo=UTC),
    )

    assert context["ticker"] == "AAPL"
    assert context["company_name"] == "Apple Inc."
    assert "Price data" in context["sources_and_disclaimer"]
    assert "does not provide financial advice" in context["sources_and_disclaimer"]
    assert "Bear" in context["scenario_table"]
    assert "MSFT" in context["peer_comparison"]


def test_render_markdown_report_includes_template_sections() -> None:
    context = report_generator.build_report_context(
        ticker="AAPL",
        price_history=sample_price_history(),
        company_profile=sample_profile(),
        valuation_table=sample_valuation_table(),
    )

    markdown = report_generator.render_markdown_report(context)

    assert "# Apple Inc. (AAPL) Equity Research Report" in markdown
    assert "## 7. Bear / Base / Bull Scenarios" in markdown
    assert "## Data Sources and Disclaimer" in markdown


def test_save_report_writes_markdown_file(tmp_path) -> None:
    output_path = report_generator.save_report("# Test\n", "aapl", output_dir=tmp_path)

    assert output_path == tmp_path / "AAPL_report.md"
    assert output_path.read_text(encoding="utf-8") == "# Test\n"


def test_format_dataframe_markdown_renders_rows() -> None:
    markdown = report_generator.format_dataframe_markdown(
        pd.DataFrame([{"Ticker": "MSFT", "Latest Close": "100.00"}])
    )

    assert "| Ticker | Latest Close |" in markdown
    assert "| MSFT | 100.00 |" in markdown


def test_generate_report_for_ticker_saves_report_when_financials_are_unavailable(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(report_generator, "get_price_history", lambda *args, **kwargs: sample_price_history())
    monkeypatch.setattr(report_generator, "get_company_profile", lambda ticker: sample_profile())
    monkeypatch.setattr(
        report_generator,
        "get_financial_statements",
        lambda *args, **kwargs: (_ for _ in ()).throw(FinancialDataError("No statements")),
    )
    monkeypatch.setattr(
        report_generator,
        "get_ttm_metrics",
        lambda ticker: (_ for _ in ()).throw(FinancialDataError("No statements")),
    )

    output_path = report_generator.generate_report_for_ticker("AAPL", output_dir=tmp_path)
    markdown = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "Financial statement data is currently unavailable" in markdown
    assert "Apple Inc. (AAPL)" in markdown
