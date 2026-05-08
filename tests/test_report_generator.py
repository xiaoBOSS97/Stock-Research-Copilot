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
    assert "Market setup" in context["executive_summary"]
    assert "Financial metrics available" in context["data_quality_notes"]
    assert "Key Assumptions" in context["valuation_assumptions"]
    assert "Target Price" in context["valuation_assumptions"]
    assert "Upside/Downside" in context["valuation_assumptions"]
    assert "Price data" in context["sources_and_disclaimer"]
    assert "does not provide financial advice" in context["sources_and_disclaimer"]
    assert "Bear" in context["scenario_table"]
    assert context["has_peer_comparison"] is True
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
    assert "## Executive Summary" in markdown
    assert "### Scenario Details" in markdown
    assert "## Data Quality Notes" in markdown
    assert "## Terms Used" not in markdown
    assert "## Bear / Base / Bull Scenarios" not in markdown
    assert "## Peer Snapshot" not in markdown
    assert "## Data Sources and Disclaimer" in markdown


def test_render_markdown_report_includes_peer_snapshot_when_supplied() -> None:
    context = report_generator.build_report_context(
        ticker="AAPL",
        price_history=sample_price_history(),
        company_profile=sample_profile(),
        valuation_table=sample_valuation_table(),
        peer_comparison="| Ticker | Latest Close |\n| --- | --- |\n| MSFT | 100.00 |",
    )

    markdown = report_generator.render_markdown_report(context)

    assert "## Peer Snapshot" in markdown
    assert "MSFT" in markdown


def test_default_report_valuation_uses_supplied_live_multiples(monkeypatch) -> None:
    monkeypatch.setattr(
        report_generator,
        "get_ttm_metrics",
        lambda ticker: {"eps": 5.0, "revenue": 1_000.0, "free_cash_flow": 100.0},
    )

    table, _ = report_generator._build_default_valuation(
        "AAPL",
        current_price=100.0,
        market_cap=10_000.0,
        shares_outstanding=100.0,
        base_pe_multiple=30.0,
        base_ps_multiple=8.0,
    )

    base_inputs = table.loc[table["scenario"] == "Base", "assumed_inputs"].iloc[0]
    assert base_inputs["pe_target_price"] == 150.0
    assert base_inputs["ps_target_price"] == 80.0


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
