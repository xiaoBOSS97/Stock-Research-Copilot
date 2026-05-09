from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.market_implied_expectations import (
    scenario_price_deviation,
    solve_implied_fcf_growth,
    summarize_market_implied_expectations,
)
from src.analysis.valuation import DcfAssumption, estimate_by_dcf


def dcf_price(growth_rate: float) -> float:
    table = estimate_by_dcf(
        [
            DcfAssumption(
                scenario=scenario,
                base_free_cash_flow=100.0,
                growth_rate=growth_rate,
                discount_rate=0.10,
                terminal_growth_rate=0.02,
                net_debt=0.0,
                shares_outstanding=10.0,
                projection_years=5,
            )
            for scenario in ["Bear", "Base", "Bull"]
        ]
    )
    return float(table.loc[table["scenario"] == "Base", "target_price"].iloc[0])


def test_solve_implied_fcf_growth_reverse_solves_growth_rate() -> None:
    current_price = dcf_price(0.05)

    result = solve_implied_fcf_growth(
        current_price=current_price,
        base_free_cash_flow=100.0,
        discount_rate=0.10,
        terminal_growth_rate=0.02,
        shares_outstanding=10.0,
        net_debt=0.0,
        projection_years=5,
    )

    assert result.status == "ok"
    assert result.implied_growth_rate == pytest.approx(0.05, abs=0.001)
    assert result.model_price == pytest.approx(current_price, abs=0.01)


def test_solve_implied_fcf_growth_handles_missing_inputs() -> None:
    result = solve_implied_fcf_growth(
        current_price=100.0,
        base_free_cash_flow=None,
        discount_rate=0.10,
        terminal_growth_rate=0.02,
        shares_outstanding=10.0,
    )

    assert result.status == "insufficient_data"
    assert "free cash flow" in result.message.lower()


def test_scenario_price_deviation_compares_current_price() -> None:
    valuation_table = pd.DataFrame(
        {
            "scenario": ["Bear", "Base", "Bull"],
            "target_price": [80.0, 100.0, 120.0],
        }
    )

    result = scenario_price_deviation(valuation_table, current_price=100.0)

    assert result.loc[0, "position"] == "above scenario"
    assert result.loc[1, "position"] == "at scenario"
    assert result.loc[2, "upside_downside"] == pytest.approx(0.20)


def test_summarize_market_implied_expectations_mentions_base_scenario() -> None:
    implied = solve_implied_fcf_growth(
        current_price=dcf_price(0.03),
        base_free_cash_flow=100.0,
        discount_rate=0.10,
        terminal_growth_rate=0.02,
        shares_outstanding=10.0,
    )
    deviation = pd.DataFrame(
        {
            "scenario": ["Base"],
            "target_price": [120.0],
            "upside_downside": [0.20],
            "position": ["below scenario"],
        }
    )

    summary = summarize_market_implied_expectations(implied, deviation)

    assert "annual FCF growth" in summary
    assert "below scenario" in summary
