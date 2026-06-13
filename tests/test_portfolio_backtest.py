from __future__ import annotations

import pytest
import pandas as pd

from src.analysis.portfolio_backtest import (
    PortfolioBacktestError,
    calculate_buy_and_hold_values,
    close_prices_from_history,
    contribution_schedule,
    normalize_weights,
    run_buy_and_hold_backtest,
)


def price_frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": values},
        index=pd.date_range("2021-01-01", periods=len(values), freq="D"),
    )


def test_normalize_weights_accepts_percent_style_values() -> None:
    weights = normalize_weights({"spy": 50, "AAPL": 30, "gld": 20})

    assert weights == {"SPY": 0.5, "AAPL": 0.3, "GLD": 0.2}


def test_normalize_weights_rejects_invalid_portfolio() -> None:
    with pytest.raises(PortfolioBacktestError, match="positive"):
        normalize_weights({"SPY": 0, "GLD": 0})

    with pytest.raises(PortfolioBacktestError, match="must not be negative"):
        normalize_weights({"SPY": -1})


def test_close_prices_from_history_aligns_common_dates() -> None:
    prices = close_prices_from_history(
        {
            "SPY": price_frame([100, 110, 120]),
            "GLD": price_frame([50, 55, 60]),
        }
    )

    assert prices.columns.tolist() == ["SPY", "GLD"]
    assert len(prices) == 3


def test_calculate_buy_and_hold_values_uses_initial_weights() -> None:
    close_prices = pd.DataFrame(
        {"SPY": [100, 120], "GLD": [50, 50]},
        index=pd.date_range("2021-01-01", periods=2, freq="D"),
    )

    portfolio_value, asset_values = calculate_buy_and_hold_values(
        close_prices,
        {"SPY": 0.5, "GLD": 0.5},
        initial_investment=10_000,
    )

    assert portfolio_value.iloc[0] == 10_000
    assert portfolio_value.iloc[-1] == 11_000
    assert asset_values.iloc[-1]["SPY"] == 6_000
    assert asset_values.iloc[-1]["GLD"] == 5_000


def test_run_buy_and_hold_backtest_returns_metrics_and_contributions() -> None:
    result = run_buy_and_hold_backtest(
        {
            "SPY": price_frame([100, 120, 110, 130]),
            "GLD": price_frame([50, 50, 60, 60]),
        },
        {"SPY": 60, "GLD": 40},
        initial_investment=10_000,
    )

    assert result.initial_investment == 10_000
    assert result.total_contributed == 10_000
    assert result.recurring_contribution == 0
    assert result.contribution_frequency == "none"
    assert result.final_value == 12_600
    assert result.profit_loss == 2_600
    assert result.total_return == pytest.approx(0.26)
    assert result.max_drawdown is not None
    assert result.annualized_return is not None
    assert result.annualized_volatility is not None
    assert result.asset_contributions["ticker"].tolist() == ["SPY", "GLD"]
    assert result.asset_contributions.loc[0, "final_value"] == 7_800
    assert result.contributions.iloc[0] == 10_000
    assert result.contributions.iloc[1:].sum() == 0


def test_contribution_schedule_supports_monthly_and_yearly() -> None:
    index = pd.date_range("2021-01-01", "2022-02-05", freq="D")

    monthly = contribution_schedule(index, "monthly")
    yearly = contribution_schedule(index, "yearly")

    assert pd.Timestamp("2021-02-01") in monthly
    assert pd.Timestamp("2022-01-01") in yearly
    assert pd.Timestamp("2021-01-01") not in monthly
    assert pd.Timestamp("2021-01-01") not in yearly


def test_run_buy_and_hold_backtest_supports_monthly_contributions() -> None:
    result = run_buy_and_hold_backtest(
        {
            "SPY": pd.DataFrame(
                {"Close": [100, 100, 100]},
                index=pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"]),
            )
        },
        {"SPY": 1},
        initial_investment=1_000,
        recurring_contribution=100,
        contribution_frequency="monthly",
    )

    assert result.total_contributed == 1_200
    assert result.final_value == 1_200
    assert result.profit_loss == 0
    assert result.total_return == 0
    assert result.contributions.tolist() == [1_000, 100, 100]
    assert result.asset_contributions.loc[0, "initial_value"] == 1_200


def test_run_buy_and_hold_backtest_requires_overlapping_prices() -> None:
    with pytest.raises(PortfolioBacktestError, match="at least two"):
        run_buy_and_hold_backtest(
            {"SPY": price_frame([100])},
            {"SPY": 1},
            initial_investment=10_000,
        )
