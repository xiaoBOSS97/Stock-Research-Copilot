from __future__ import annotations

import pandas as pd

from src.app.streamlit_app import (
    build_valuation_table,
    default_period_index,
    format_optional_percent,
    grouped_performance_metric_specs,
    is_current_report_preview,
    metric_card_specs,
    metrics_to_display_frame,
    market_implied_to_display_frame,
    market_implied_status_caption,
    performance_metric_specs,
    parse_peer_input,
    statement_row,
    valuation_input_defaults,
    valuation_to_display_frame,
)


def test_parse_peer_input_normalizes_symbols() -> None:
    assert parse_peer_input(" msft, googl ,, meta ") == ["MSFT", "GOOGL", "META"]


def test_default_period_index_uses_settings_value() -> None:
    assert default_period_index({"default_period": "3y"}) == 1
    assert default_period_index({"default_period": "bad"}) == 2


def test_metric_card_specs_include_native_help_for_every_metric() -> None:
    cards = metric_card_specs(
        {
            "latest_close": 100.0,
            "RSI": 55.0,
            "annualized_volatility": 0.2,
        },
        {
            "revenue_growth_yoy": 0.1,
            "net_margin": 0.2,
            "fcf_margin": 0.15,
        },
    )

    assert [label for label, _, _ in cards] == [
        "Close",
        "RSI",
        "Volatility",
        "Revenue Growth",
        "Net Margin",
        "FCF Margin",
    ]
    assert all(help_text != "Term explanation is not available yet." for _, _, help_text in cards)


def test_performance_metric_specs_include_native_help_for_every_metric() -> None:
    metrics = {
        "revenue_growth_yoy": 0.1,
        "net_income_growth_yoy": 0.2,
        "revenue_cagr": 0.08,
        "gross_margin": 0.5,
        "operating_margin": 0.3,
        "net_margin": 0.2,
        "fcf_margin": 0.15,
        "roe": 0.4,
        "roa": 0.2,
        "roic": 0.25,
        "debt_to_equity": 1.1,
        "current_ratio": 1.5,
        "interest_coverage": 12.0,
        "free_cash_flow": 100_000_000_000,
    }

    specs = performance_metric_specs(metrics)

    assert len(specs) == len(metrics)
    assert all(help_text != "Term explanation is not available yet." for _, _, help_text in specs)


def test_grouped_performance_metric_specs_groups_core_metrics() -> None:
    grouped = grouped_performance_metric_specs(
        {
            "revenue_growth_yoy": 0.1,
            "net_margin": 0.2,
            "roe": 0.4,
            "current_ratio": 1.5,
        }
    )

    group_names = [name for name, _ in grouped]

    assert group_names == ["Growth", "Profitability", "Returns", "Balance Sheet"]


def test_is_current_report_preview_requires_executive_summary_and_schema() -> None:
    assert is_current_report_preview("## Executive Summary\n", "AAPL", "AAPL", "market-implied-v1")
    assert not is_current_report_preview("## One-line Summary\n", "AAPL", "AAPL", "market-implied-v1")
    assert not is_current_report_preview("## Executive Summary\n", "AAPL", "MSFT", "market-implied-v1")
    assert not is_current_report_preview("## Executive Summary\n", "AAPL", "AAPL", "old")


def test_format_optional_percent_handles_missing_values() -> None:
    assert format_optional_percent(None) == "N/A"
    assert format_optional_percent(0.1234) == "12.3%"


def test_statement_row_finds_candidate_case_insensitively() -> None:
    statement = pd.DataFrame({"2025-12-31": [123.0]}, index=["Total Revenue"])

    result = statement_row(statement, ("total revenue",))

    assert result is not None
    assert result.iloc[0] == 123.0


def test_build_valuation_table_returns_selected_method() -> None:
    table = build_valuation_table(
        method="PE",
        latest_close=100.0,
        market_cap=1_000.0,
        shares_outstanding=None,
        base_eps=5.0,
        base_pe=20.0,
        base_revenue_billions=10.0,
        base_ps=2.0,
    )

    assert table["method"].unique().tolist() == ["PE"]
    assert table["scenario"].tolist() == ["Bear", "Base", "Bull"]


def test_build_valuation_table_supports_dcf() -> None:
    table = build_valuation_table(
        method="DCF",
        latest_close=100.0,
        market_cap=1_000.0,
        shares_outstanding=None,
        base_eps=5.0,
        base_pe=20.0,
        base_revenue_billions=10.0,
        base_ps=2.0,
        base_free_cash_flow_billions=1.0,
        dcf_growth_rate=0.03,
        dcf_discount_rate=0.10,
        dcf_terminal_growth_rate=0.02,
        net_debt_billions=0.0,
    )

    assert table["method"].unique().tolist() == ["DCF"]
    assert table["status"].tolist() == ["ok", "ok", "ok"]


def test_metrics_to_display_frame_humanizes_labels_and_values() -> None:
    frame = metrics_to_display_frame({"net_margin": 0.25, "free_cash_flow": 100_000_000_000})

    assert frame.iloc[0]["Metric"] == "Net Margin"
    assert frame.iloc[0]["Value"] == "25.0%"
    assert frame.iloc[1]["Value"] == "100.00B"


def test_valuation_to_display_frame_hides_raw_assumptions() -> None:
    table = build_valuation_table(
        method="PE",
        latest_close=100.0,
        market_cap=1_000.0,
        shares_outstanding=None,
        base_eps=5.0,
        base_pe=20.0,
        base_revenue_billions=10.0,
        base_ps=2.0,
    )

    display = valuation_to_display_frame(table, current_price=100.0)

    assert "Assumed Inputs" not in display.columns
    assert "Target Price" in display.columns
    assert "Upside/Downside" in display.columns
    assert display.loc[1, "Target Price"] == "100.00"
    assert display.loc[1, "Upside/Downside"] == "0.0%"


def test_market_implied_to_display_frame_formats_values() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario": "Base",
                "target_price": 120.0,
                "price_gap": 20.0,
                "upside_downside": 0.2,
                "position": "below scenario",
            }
        ]
    )

    display = market_implied_to_display_frame(frame)

    assert display.loc[0, "Target Price"] == "120.00"
    assert display.loc[0, "Price Gap"] == "20.00"
    assert display.loc[0, "Upside/Downside"] == "20.0%"


def test_market_implied_status_caption_explains_status() -> None:
    assert "Solved" in market_implied_status_caption("ok")
    assert "outside" in market_implied_status_caption("out_of_bounds")
    assert "Solver needs" in market_implied_status_caption("insufficient_data")


def test_valuation_input_defaults_use_live_profile_fields() -> None:
    defaults = valuation_input_defaults(
        {
            "forwardEps": 12.0,
            "forwardPE": 25.0,
            "totalRevenue": 500_000_000_000,
            "priceToSalesTrailing12Months": 8.0,
            "freeCashflow": 120_000_000_000,
            "sharesOutstanding": 15_000_000_000,
            "totalDebt": 110_000_000_000,
            "totalCash": 70_000_000_000,
        },
        {},
        {},
    )

    assert defaults["base_eps"] == 12.0
    assert defaults["base_pe"] == 25.0
    assert defaults["base_revenue_billions"] == 500.0
    assert defaults["base_ps"] == 8.0
    assert defaults["base_free_cash_flow_billions"] == 120.0
    assert defaults["shares_outstanding"] == 15_000_000_000
    assert defaults["net_debt_billions"] == 40.0


def test_valuation_input_defaults_fall_back_to_statements_and_fixed_values() -> None:
    defaults = valuation_input_defaults(
        {},
        {
            "income_statement": pd.DataFrame(
                {"2025-12-31": [300_000_000_000]},
                index=["Total Revenue"],
            )
        },
        {"free_cash_flow": 90_000_000_000},
    )

    assert defaults["base_eps"] == 10.0
    assert defaults["base_pe"] == 22.0
    assert defaults["base_revenue_billions"] == 300.0
    assert defaults["base_ps"] == 6.0
    assert defaults["base_free_cash_flow_billions"] == 90.0
