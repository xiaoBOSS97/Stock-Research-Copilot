"""Market-implied expectation helpers for valuation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.analysis.valuation import DcfAssumption, estimate_by_dcf


@dataclass(frozen=True)
class ImpliedGrowthResult:
    """Reverse-solved DCF growth expectation."""

    implied_growth_rate: float | None
    current_price: float | None
    model_price: float | None
    status: str
    message: str

    def to_dict(self) -> dict[str, float | str | None]:
        """Return a stable dictionary representation."""

        return {
            "implied_growth_rate": self.implied_growth_rate,
            "current_price": self.current_price,
            "model_price": self.model_price,
            "status": self.status,
            "message": self.message,
        }


def solve_implied_fcf_growth(
    current_price: float | None,
    base_free_cash_flow: float | None,
    discount_rate: float | None,
    terminal_growth_rate: float | None,
    shares_outstanding: float | None,
    *,
    net_debt: float | None = 0.0,
    projection_years: int = 5,
    lower_bound: float = -0.50,
    upper_bound: float = 0.50,
    tolerance: float = 0.0001,
    max_iterations: int = 80,
) -> ImpliedGrowthResult:
    """Estimate the FCF growth rate implied by the current share price."""

    if _invalid(current_price) or current_price <= 0:
        return _failed_result(current_price, "Current price must be positive.")
    if _invalid(base_free_cash_flow) or base_free_cash_flow <= 0:
        return _failed_result(current_price, "Base free cash flow must be positive.")
    if _invalid(discount_rate) or _invalid(terminal_growth_rate):
        return _failed_result(current_price, "Discount rate and terminal growth are required.")
    if discount_rate <= terminal_growth_rate:
        return _failed_result(current_price, "Discount rate must be above terminal growth.")
    if _invalid(shares_outstanding) or shares_outstanding <= 0:
        return _failed_result(current_price, "Shares outstanding must be positive.")
    if projection_years <= 0:
        return _failed_result(current_price, "Projection years must be positive.")

    low_price = _dcf_price_for_growth(
        lower_bound,
        base_free_cash_flow,
        discount_rate,
        terminal_growth_rate,
        net_debt,
        shares_outstanding,
        projection_years,
    )
    high_price = _dcf_price_for_growth(
        upper_bound,
        base_free_cash_flow,
        discount_rate,
        terminal_growth_rate,
        net_debt,
        shares_outstanding,
        projection_years,
    )
    if low_price is None or high_price is None:
        return _failed_result(current_price, "DCF inputs are insufficient.")
    if current_price < low_price or current_price > high_price:
        return ImpliedGrowthResult(
            implied_growth_rate=None,
            current_price=float(current_price),
            model_price=None,
            status="out_of_bounds",
            message=(
                "Current price is outside the solvable growth range. "
                f"At {lower_bound:.0%} to {upper_bound:.0%} FCF growth, model prices range "
                f"from {low_price:.2f} to {high_price:.2f}."
            ),
        )

    low = lower_bound
    high = upper_bound
    mid = (low + high) / 2
    mid_price = None
    for _ in range(max_iterations):
        mid = (low + high) / 2
        mid_price = _dcf_price_for_growth(
            mid,
            base_free_cash_flow,
            discount_rate,
            terminal_growth_rate,
            net_debt,
            shares_outstanding,
            projection_years,
        )
        if mid_price is None:
            return _failed_result(current_price, "DCF inputs are insufficient.")
        if abs(mid_price - current_price) <= tolerance:
            break
        if mid_price < current_price:
            low = mid
        else:
            high = mid

    return ImpliedGrowthResult(
        implied_growth_rate=float(mid),
        current_price=float(current_price),
        model_price=None if mid_price is None else float(mid_price),
        status="ok",
        message=(
            "Current price is approximately consistent with "
            f"{mid:.1%} annual FCF growth in this simplified DCF setup."
        ),
    )


def scenario_price_deviation(
    valuation_table: pd.DataFrame,
    current_price: float | None,
) -> pd.DataFrame:
    """Compare current price with scenario target prices."""

    if valuation_table.empty:
        return pd.DataFrame(columns=["scenario", "target_price", "price_gap", "upside_downside", "position"])

    rows = []
    for _, row in valuation_table.iterrows():
        target_price = _to_float(row.get("target_price"))
        gap = None
        upside_downside = None
        position = "N/A"
        if target_price is not None and current_price is not None and current_price > 0:
            gap = target_price - current_price
            upside_downside = target_price / current_price - 1
            if current_price < target_price:
                position = "below scenario"
            elif current_price > target_price:
                position = "above scenario"
            else:
                position = "at scenario"
        rows.append(
            {
                "scenario": row.get("scenario", "N/A"),
                "target_price": target_price,
                "price_gap": gap,
                "upside_downside": upside_downside,
                "position": position,
            }
        )
    return pd.DataFrame(rows)


def summarize_market_implied_expectations(
    implied_growth: ImpliedGrowthResult,
    scenario_deviation: pd.DataFrame,
) -> str:
    """Build a concise report/dashboard summary."""

    if implied_growth.status == "ok":
        growth_text = implied_growth.message
    else:
        growth_text = f"Implied growth could not be solved: {implied_growth.message}"

    if scenario_deviation.empty or "scenario" not in scenario_deviation:
        return growth_text

    base_rows = scenario_deviation[scenario_deviation["scenario"] == "Base"]
    if base_rows.empty:
        return growth_text
    base_position = base_rows.iloc[0].get("position", "N/A")
    base_upside = base_rows.iloc[0].get("upside_downside")
    if base_upside is None or pd.isna(base_upside):
        return growth_text
    return f"{growth_text} Current price is {base_position} with {base_upside:.1%} upside/downside to the base scenario."


def _dcf_price_for_growth(
    growth_rate: float,
    base_free_cash_flow: float,
    discount_rate: float,
    terminal_growth_rate: float,
    net_debt: float | None,
    shares_outstanding: float,
    projection_years: int,
) -> float | None:
    frame = estimate_by_dcf(
        [
            DcfAssumption(
                scenario="Bear",
                base_free_cash_flow=base_free_cash_flow,
                growth_rate=growth_rate,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth_rate,
                net_debt=net_debt,
                shares_outstanding=shares_outstanding,
                projection_years=projection_years,
            ),
            DcfAssumption(
                scenario="Base",
                base_free_cash_flow=base_free_cash_flow,
                growth_rate=growth_rate,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth_rate,
                net_debt=net_debt,
                shares_outstanding=shares_outstanding,
                projection_years=projection_years,
            ),
            DcfAssumption(
                scenario="Bull",
                base_free_cash_flow=base_free_cash_flow,
                growth_rate=growth_rate,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth_rate,
                net_debt=net_debt,
                shares_outstanding=shares_outstanding,
                projection_years=projection_years,
            ),
        ]
    )
    value = frame.loc[frame["scenario"] == "Base", "target_price"].iloc[0]
    return _to_float(value)


def _failed_result(current_price: float | None, message: str) -> ImpliedGrowthResult:
    return ImpliedGrowthResult(
        implied_growth_rate=None,
        current_price=_to_float(current_price),
        model_price=None,
        status="insufficient_data",
        message=message,
    )


def _invalid(value: Any) -> bool:
    return value is None or pd.isna(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)
