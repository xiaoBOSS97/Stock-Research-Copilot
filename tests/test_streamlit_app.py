from __future__ import annotations

import pandas as pd

from src.app.streamlit_app import (
    build_valuation_table,
    format_optional_percent,
    parse_peer_input,
    statement_row,
)


def test_parse_peer_input_normalizes_symbols() -> None:
    assert parse_peer_input(" msft, googl ,, meta ") == ["MSFT", "GOOGL", "META"]


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
        base_eps=5.0,
        base_pe=20.0,
        base_revenue_billions=10.0,
        base_ps=2.0,
    )

    assert table["method"].unique().tolist() == ["PE"]
    assert table["scenario"].tolist() == ["Bear", "Base", "Bull"]
