from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.technical_analysis import (
    add_technical_indicators,
    add_moving_averages,
    calculate_technical_metrics,
    calculate_max_drawdown,
    calculate_rsi,
    calculate_volatility,
    generate_technical_summary,
)


@pytest.fixture
def price_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Close": [
                100,
                101,
                102,
                101,
                103,
                105,
                104,
                106,
                108,
                107,
                109,
                111,
                110,
                112,
                114,
                113,
                115,
                117,
                116,
                118,
            ]
        },
        index=pd.date_range("2025-01-01", periods=20, freq="D"),
    )


def test_moving_average_columns_exist(price_data: pd.DataFrame) -> None:
    result = add_moving_averages(price_data, windows=(3, 5))

    assert "MA3" in result.columns
    assert "MA5" in result.columns
    assert result["MA3"].iloc[-1] == pytest.approx((117 + 116 + 118) / 3)


def test_rsi_range_between_0_and_100(price_data: pd.DataFrame) -> None:
    rsi = calculate_rsi(price_data)

    assert rsi.between(0, 100).all()
    assert rsi.name == "RSI"


def test_rsi_is_neutral_for_flat_series() -> None:
    flat_data = pd.DataFrame({"Close": [100.0] * 20})

    rsi = calculate_rsi(flat_data)

    assert rsi.iloc[-1] == pytest.approx(50)


def test_rsi_reaches_extreme_for_one_direction_series() -> None:
    rising = pd.DataFrame({"Close": list(range(1, 21))})
    falling = pd.DataFrame({"Close": list(range(20, 0, -1))})

    assert calculate_rsi(rising).iloc[-1] == pytest.approx(100)
    assert calculate_rsi(falling).iloc[-1] == pytest.approx(0)


def test_max_drawdown_is_non_positive(price_data: pd.DataFrame) -> None:
    max_drawdown = calculate_max_drawdown(price_data)

    assert max_drawdown <= 0


def test_volatility_is_non_negative(price_data: pd.DataFrame) -> None:
    volatility = calculate_volatility(price_data)

    assert volatility >= 0


def test_add_technical_indicators_adds_mvp_columns(price_data: pd.DataFrame) -> None:
    result = add_technical_indicators(price_data)

    assert {"MA20", "MA50", "MA200", "RSI", "annualized_volatility", "max_drawdown"}.issubset(
        result.columns
    )
    assert result["annualized_volatility"].nunique() == 1
    assert result["max_drawdown"].nunique() == 1


def test_calculate_technical_metrics_returns_dashboard_shape(price_data: pd.DataFrame) -> None:
    metrics = calculate_technical_metrics(price_data)

    assert {
        "latest_close",
        "MA20",
        "MA50",
        "MA200",
        "RSI",
        "annualized_volatility",
        "max_drawdown",
        "trend",
        "momentum",
    } == set(metrics)
    assert metrics["latest_close"] == pytest.approx(118)
    assert metrics["trend"] in {"above_ma200", "below_ma200"}
    assert metrics["momentum"] in {"overheated", "neutral", "oversold"}


def test_generate_technical_summary_includes_disclaimer_language(price_data: pd.DataFrame) -> None:
    summary = generate_technical_summary(price_data)

    assert "RSI" in summary
    assert "do not replace fundamental analysis" in summary


def test_empty_price_data_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_max_drawdown(pd.DataFrame(columns=["Close"]))


def test_missing_close_column_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Close"):
        calculate_volatility(pd.DataFrame({"Open": [1, 2, 3]}))


def test_invalid_windows_raise_clear_errors(price_data: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="positive"):
        add_moving_averages(price_data, windows=(0,))

    with pytest.raises(ValueError, match="positive"):
        calculate_rsi(price_data, window=0)

    with pytest.raises(ValueError, match="positive"):
        calculate_volatility(price_data, trading_days=0)
