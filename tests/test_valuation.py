from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.valuation import (
    ScenarioAssumption,
    blended_valuation,
    build_scenarios,
    estimate_by_pe,
    estimate_by_ps,
    summarize_valuation,
)


@pytest.fixture
def assumptions() -> list[ScenarioAssumption]:
    return build_scenarios(
        base_forward_eps=10.0,
        base_pe_multiple=20.0,
        base_forward_revenue=100_000.0,
        base_ps_multiple=5.0,
        shares_outstanding=1_000.0,
    )


def test_pe_valuation_formula(assumptions: list[ScenarioAssumption]) -> None:
    result = estimate_by_pe(assumptions)

    base_row = result[result["scenario"] == "Base"].iloc[0]
    assert base_row["method"] == "PE"
    assert base_row["target_price"] == pytest.approx(200.0)
    assert base_row["status"] == "ok"


def test_ps_valuation_formula(assumptions: list[ScenarioAssumption]) -> None:
    result = estimate_by_ps(assumptions)

    base_row = result[result["scenario"] == "Base"].iloc[0]
    assert base_row["method"] == "PS"
    assert base_row["target_price"] == pytest.approx(500.0)
    assert base_row["status"] == "ok"


def test_scenario_order_bear_base_bull(assumptions: list[ScenarioAssumption]) -> None:
    result = estimate_by_pe(assumptions)

    assert result["scenario"].tolist() == ["Bear", "Base", "Bull"]
    assert result["target_price"].tolist() == sorted(result["target_price"].tolist())


def test_blended_valuation_averages_available_methods(
    assumptions: list[ScenarioAssumption],
) -> None:
    result = blended_valuation(assumptions)

    base_row = result[result["scenario"] == "Base"].iloc[0]
    assert base_row["method"] == "blended"
    assert base_row["target_price"] == pytest.approx(350.0)


def test_missing_inputs_return_insufficient_data() -> None:
    assumptions = build_scenarios(
        base_forward_eps=None,
        base_pe_multiple=20.0,
        base_forward_revenue=100_000.0,
        base_ps_multiple=None,
        shares_outstanding=1_000.0,
    )

    pe = estimate_by_pe(assumptions)
    ps = estimate_by_ps(assumptions)

    assert set(pe["status"]) == {"insufficient_data"}
    assert set(ps["status"]) == {"insufficient_data"}
    assert pe["target_price"].isna().all()


def test_blended_valuation_uses_available_method_when_one_is_missing() -> None:
    assumptions = build_scenarios(
        base_forward_eps=10.0,
        base_pe_multiple=20.0,
        base_forward_revenue=None,
        base_ps_multiple=5.0,
        shares_outstanding=1_000.0,
    )

    result = blended_valuation(assumptions)
    base_row = result[result["scenario"] == "Base"].iloc[0]

    assert base_row["target_price"] == pytest.approx(200.0)
    assert base_row["status"] == "ok"


def test_invalid_custom_scenario_order_raises() -> None:
    with pytest.raises(ValueError, match="Bear, Base, Bull"):
        build_scenarios(
            custom_assumptions=[
                ScenarioAssumption("Base", 1.0, 2.0, 3.0, 4.0, 5.0),
                ScenarioAssumption("Bear", 1.0, 2.0, 3.0, 4.0, 5.0),
                ScenarioAssumption("Bull", 1.0, 2.0, 3.0, 4.0, 5.0),
            ]
        )


def test_invalid_blended_weights_raise(assumptions: list[ScenarioAssumption]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        blended_valuation(assumptions, pe_weight=-1)

    with pytest.raises(ValueError, match="positive"):
        blended_valuation(assumptions, pe_weight=0, ps_weight=0)


def test_summarize_valuation_compares_current_price(
    assumptions: list[ScenarioAssumption],
) -> None:
    summary = summarize_valuation(estimate_by_pe(assumptions), current_price=200.0)

    assert "Scenario valuation range" in summary
    assert "Current price is within" in summary
    assert "not financial advice" in summary


def test_summarize_valuation_handles_empty_prices() -> None:
    table = pd.DataFrame(
        {
            "scenario": ["Bear", "Base", "Bull"],
            "method": ["PE", "PE", "PE"],
            "target_price": [None, None, None],
        }
    )

    summary = summarize_valuation(table)

    assert "could not be calculated" in summary
