from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.financial_ratios import (
    calculate_financial_metrics,
    calculate_free_cash_flow,
    calculate_growth_rates,
    calculate_leverage,
    calculate_margins,
    calculate_returns,
    generate_financial_summary,
)


@pytest.fixture
def statements() -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31"])
    return {
        "income_statement": pd.DataFrame(
            {
                dates[0]: [120.0, 60.0, 36.0, 30.0, 6.0, 30.0, -3.0],
                dates[1]: [100.0, 45.0, 25.0, 20.0, 5.0, 25.0, -4.0],
                dates[2]: [80.0, 32.0, 16.0, 10.0, 4.0, 20.0, -5.0],
            },
            index=[
                "Total Revenue",
                "Gross Profit",
                "Operating Income",
                "Net Income",
                "Tax Provision",
                "Pretax Income",
                "Interest Expense",
            ],
        ),
        "balance_sheet": pd.DataFrame(
            {
                dates[0]: [300.0, 100.0, 80.0, 150.0, 75.0, 20.0],
                dates[1]: [260.0, 90.0, 70.0, 120.0, 60.0, 15.0],
                dates[2]: [220.0, 80.0, 50.0, 100.0, 50.0, 10.0],
            },
            index=[
                "Total Assets",
                "Stockholders Equity",
                "Total Debt",
                "Current Assets",
                "Current Liabilities",
                "Cash And Cash Equivalents",
            ],
        ),
        "cash_flow": pd.DataFrame(
            {
                dates[0]: [40.0, -12.0],
                dates[1]: [30.0, -10.0],
                dates[2]: [22.0, -8.0],
            },
            index=["Operating Cash Flow", "Capital Expenditure"],
        ),
    }


def test_growth_rate_calculation(statements: dict[str, pd.DataFrame]) -> None:
    metrics = calculate_growth_rates(statements["income_statement"])

    assert metrics["revenue_growth_yoy"] == pytest.approx(0.2)
    assert metrics["net_income_growth_yoy"] == pytest.approx(0.5)
    assert metrics["revenue_cagr"] == pytest.approx((120 / 80) ** 0.5 - 1)


def test_margin_calculation(statements: dict[str, pd.DataFrame]) -> None:
    metrics = calculate_margins(statements["income_statement"], statements["cash_flow"])

    assert metrics["gross_margin"] == pytest.approx(0.5)
    assert metrics["operating_margin"] == pytest.approx(0.3)
    assert metrics["net_margin"] == pytest.approx(0.25)
    assert metrics["fcf_margin"] == pytest.approx((40 - 12) / 120)


def test_free_cash_flow_calculation(statements: dict[str, pd.DataFrame]) -> None:
    fcf = calculate_free_cash_flow(statements["cash_flow"])

    assert fcf.iloc[0] == pytest.approx(28)
    assert fcf.iloc[1] == pytest.approx(20)


def test_returns_calculation(statements: dict[str, pd.DataFrame]) -> None:
    metrics = calculate_returns(statements["income_statement"], statements["balance_sheet"])

    assert metrics["roe"] == pytest.approx(0.3)
    assert metrics["roa"] == pytest.approx(0.1)
    assert metrics["roic"] == pytest.approx((36 * 0.8) / (80 + 100 - 20))


def test_leverage_calculation(statements: dict[str, pd.DataFrame]) -> None:
    metrics = calculate_leverage(statements["income_statement"], statements["balance_sheet"])

    assert metrics["debt_to_equity"] == pytest.approx(0.8)
    assert metrics["current_ratio"] == pytest.approx(2.0)
    assert metrics["interest_coverage"] == pytest.approx(12.0)


def test_calculate_financial_metrics_combines_mvp_outputs(statements: dict[str, pd.DataFrame]) -> None:
    metrics = calculate_financial_metrics(statements)

    assert metrics["revenue_growth_yoy"] == pytest.approx(0.2)
    assert metrics["net_margin"] == pytest.approx(0.25)
    assert metrics["free_cash_flow"] == pytest.approx(28)
    assert metrics["debt_to_equity"] == pytest.approx(0.8)


def test_empty_financials_return_none_or_raise_clear_error() -> None:
    empty_income = pd.DataFrame()
    empty_cash_flow = pd.DataFrame()

    assert calculate_growth_rates(empty_income)["revenue_growth_yoy"] is None
    assert calculate_margins(empty_income)["net_margin"] is None
    with pytest.raises(ValueError, match="operating cash flow"):
        calculate_free_cash_flow(empty_cash_flow)


def test_composite_metrics_tolerate_missing_cash_flow_rows(
    statements: dict[str, pd.DataFrame],
) -> None:
    incomplete = statements | {"cash_flow": pd.DataFrame({"2025-12-31": [1.0]}, index=["Other"])}

    metrics = calculate_financial_metrics(incomplete)

    assert metrics["fcf_margin"] is None
    assert metrics["free_cash_flow"] is None


def test_generate_financial_summary_uses_rule_based_language(
    statements: dict[str, pd.DataFrame],
) -> None:
    summary = generate_financial_summary(calculate_financial_metrics(statements))

    assert "Revenue growth" in summary
    assert "Net margin" in summary
    assert "Free cash flow" in summary
