from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.valuation import (
    DcfAssumption,
    ScenarioAssumption,
    blended_valuation,
    blended_valuation_with_dcf,
    build_dcf_scenarios,
    build_scenarios,
    estimate_by_dcf,
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


def test_dcf_valuation_formula() -> None:
    dcf_assumptions = [
        DcfAssumption(
            scenario=scenario,
            base_free_cash_flow=100.0,
            growth_rate=0.0,
            discount_rate=0.10,
            terminal_growth_rate=0.02,
            net_debt=0.0,
            shares_outstanding=10.0,
            projection_years=2,
        )
        for scenario in ["Bear", "Base", "Bull"]
    ]

    result = estimate_by_dcf(dcf_assumptions)
    expected_terminal_value = 100 * 1.02 / (0.10 - 0.02)
    expected_enterprise_value = 100 / 1.10 + 100 / (1.10**2) + expected_terminal_value / (1.10**2)

    assert result.loc[1, "method"] == "DCF"
    assert result.loc[1, "target_price"] == pytest.approx(expected_enterprise_value / 10)


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


def test_build_dcf_scenarios_order_and_values() -> None:
    dcf_assumptions = build_dcf_scenarios(
        base_free_cash_flow=100_000.0,
        base_growth_rate=0.05,
        discount_rate=0.10,
        terminal_growth_rate=0.025,
        net_debt=10_000.0,
        shares_outstanding=1_000.0,
    )

    assert [assumption.scenario for assumption in dcf_assumptions] == ["Bear", "Base", "Bull"]
    assert dcf_assumptions[0].growth_rate < dcf_assumptions[1].growth_rate < dcf_assumptions[2].growth_rate
    assert dcf_assumptions[0].discount_rate > dcf_assumptions[1].discount_rate > dcf_assumptions[2].discount_rate


def test_dcf_missing_inputs_return_insufficient_data() -> None:
    dcf_assumptions = build_dcf_scenarios(
        base_free_cash_flow=None,
        discount_rate=0.10,
        terminal_growth_rate=0.02,
        shares_outstanding=1_000.0,
    )

    result = estimate_by_dcf(dcf_assumptions)

    assert set(result["status"]) == {"insufficient_data"}
    assert result["target_price"].isna().all()


def test_dcf_requires_discount_rate_above_terminal_growth() -> None:
    dcf_assumptions = build_dcf_scenarios(
        base_free_cash_flow=100_000.0,
        discount_rate=0.02,
        terminal_growth_rate=0.03,
        shares_outstanding=1_000.0,
    )

    result = estimate_by_dcf(dcf_assumptions)

    assert result.loc[result["scenario"] == "Base", "status"].iloc[0] == "insufficient_data"


def test_blended_valuation_with_dcf_combines_available_methods(
    assumptions: list[ScenarioAssumption],
) -> None:
    dcf_assumptions = build_dcf_scenarios(
        base_free_cash_flow=10_000.0,
        discount_rate=0.10,
        terminal_growth_rate=0.02,
        shares_outstanding=1_000.0,
        base_growth_rate=0.03,
    )

    result = blended_valuation_with_dcf(
        assumptions,
        dcf_assumptions=dcf_assumptions,
        pe_weight=1.0,
        ps_weight=0.0,
        dcf_weight=0.0,
    )
    base_row = result[result["scenario"] == "Base"].iloc[0]

    assert base_row["target_price"] == pytest.approx(200.0)
    assert "dcf_target_price" in base_row["assumed_inputs"]


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
