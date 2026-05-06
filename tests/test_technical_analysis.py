from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.technical_analysis import (
    add_moving_averages,
    calculate_max_drawdown,
    calculate_rsi,
    calculate_volatility,
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


def test_max_drawdown_is_non_positive(price_data: pd.DataFrame) -> None:
    max_drawdown = calculate_max_drawdown(price_data)

    assert max_drawdown <= 0


def test_volatility_is_non_negative(price_data: pd.DataFrame) -> None:
    volatility = calculate_volatility(price_data)

    assert volatility >= 0


def test_empty_price_data_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_max_drawdown(pd.DataFrame(columns=["Close"]))
