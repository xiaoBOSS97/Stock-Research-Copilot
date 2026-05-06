"""Scenario-based valuation helpers.

The MVP uses simple relative valuation methods. These functions produce
educational Bear/Base/Bull ranges from explicit assumptions; they do not make
investment recommendations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

import pandas as pd


SCENARIO_ORDER = ("Bear", "Base", "Bull")


@dataclass(frozen=True)
class ScenarioAssumption:
    """Inputs used by PE and P/S valuation scenarios."""

    scenario: str
    forward_eps: float | None
    pe_multiple: float | None
    forward_revenue: float | None
    ps_multiple: float | None
    shares_outstanding: float | None


def build_scenarios(
    base_forward_eps: float | None = None,
    base_pe_multiple: float | None = None,
    base_forward_revenue: float | None = None,
    base_ps_multiple: float | None = None,
    shares_outstanding: float | None = None,
    custom_assumptions: Iterable[ScenarioAssumption] | None = None,
) -> list[ScenarioAssumption]:
    """Build Bear/Base/Bull assumptions from base inputs or custom scenarios."""

    if custom_assumptions is not None:
        assumptions = list(custom_assumptions)
        _validate_scenarios(assumptions)
        return assumptions

    assumptions = [
        ScenarioAssumption(
            scenario="Bear",
            forward_eps=_scale(base_forward_eps, 0.85),
            pe_multiple=_scale(base_pe_multiple, 0.75),
            forward_revenue=_scale(base_forward_revenue, 0.90),
            ps_multiple=_scale(base_ps_multiple, 0.75),
            shares_outstanding=shares_outstanding,
        ),
        ScenarioAssumption(
            scenario="Base",
            forward_eps=base_forward_eps,
            pe_multiple=base_pe_multiple,
            forward_revenue=base_forward_revenue,
            ps_multiple=base_ps_multiple,
            shares_outstanding=shares_outstanding,
        ),
        ScenarioAssumption(
            scenario="Bull",
            forward_eps=_scale(base_forward_eps, 1.15),
            pe_multiple=_scale(base_pe_multiple, 1.25),
            forward_revenue=_scale(base_forward_revenue, 1.10),
            ps_multiple=_scale(base_ps_multiple, 1.25),
            shares_outstanding=shares_outstanding,
        ),
    ]
    _validate_scenarios(assumptions)
    return assumptions


def estimate_by_pe(assumptions: Iterable[ScenarioAssumption]) -> pd.DataFrame:
    """Estimate target price as ``forward_eps * pe_multiple`` for each scenario."""

    rows = []
    for assumption in assumptions:
        target_price = _multiply(assumption.forward_eps, assumption.pe_multiple)
        rows.append(
            _valuation_row(
                assumption=assumption,
                method="PE",
                target_price=target_price,
                required_fields=("forward_eps", "pe_multiple"),
            )
        )
    return _ordered_frame(rows)


def estimate_by_ps(assumptions: Iterable[ScenarioAssumption]) -> pd.DataFrame:
    """Estimate target price from revenue, P/S multiple, and shares outstanding."""

    rows = []
    for assumption in assumptions:
        equity_value = _multiply(assumption.forward_revenue, assumption.ps_multiple)
        target_price = _safe_divide(equity_value, assumption.shares_outstanding)
        rows.append(
            _valuation_row(
                assumption=assumption,
                method="PS",
                target_price=target_price,
                required_fields=("forward_revenue", "ps_multiple", "shares_outstanding"),
            )
        )
    return _ordered_frame(rows)


def blended_valuation(
    assumptions: Iterable[ScenarioAssumption],
    pe_weight: float = 0.5,
    ps_weight: float = 0.5,
) -> pd.DataFrame:
    """Blend PE and P/S valuation outputs into one scenario table."""

    if pe_weight < 0 or ps_weight < 0:
        raise ValueError("Valuation weights must be non-negative.")
    total_weight = pe_weight + ps_weight
    if total_weight == 0:
        raise ValueError("At least one valuation weight must be positive.")

    pe = estimate_by_pe(assumptions).set_index("scenario")
    ps = estimate_by_ps(assumptions).set_index("scenario")

    rows = []
    for scenario in SCENARIO_ORDER:
        pe_price = pe.loc[scenario, "target_price"] if scenario in pe.index else None
        ps_price = ps.loc[scenario, "target_price"] if scenario in ps.index else None
        weighted_prices = []
        if pd.notna(pe_price):
            weighted_prices.append((float(pe_price), pe_weight))
        if pd.notna(ps_price):
            weighted_prices.append((float(ps_price), ps_weight))

        if weighted_prices:
            available_weight = sum(weight for _, weight in weighted_prices)
            target_price = sum(price * weight for price, weight in weighted_prices) / available_weight
            status = "ok"
        else:
            target_price = None
            status = "insufficient_data"

        rows.append(
            {
                "scenario": scenario,
                "method": "blended",
                "assumed_inputs": {
                    "pe_weight": pe_weight / total_weight,
                    "ps_weight": ps_weight / total_weight,
                    "pe_target_price": None if pd.isna(pe_price) else float(pe_price),
                    "ps_target_price": None if pd.isna(ps_price) else float(ps_price),
                },
                "target_price": target_price,
                "status": status,
            }
        )

    return _ordered_frame(rows)


def summarize_valuation(valuation_table: pd.DataFrame, current_price: float | None = None) -> str:
    """Generate a disclaimer-safe valuation summary from a scenario table."""

    valid_prices = pd.to_numeric(valuation_table["target_price"], errors="coerce").dropna()
    if valid_prices.empty:
        return (
            "Valuation scenarios could not be calculated because required assumptions are missing. "
            "This is an educational model, not financial advice."
        )

    low = float(valid_prices.min())
    high = float(valid_prices.max())
    summary = f"Scenario valuation range is {low:.2f} to {high:.2f} based on the stated assumptions."
    if current_price is not None and current_price > 0:
        if current_price < low:
            comparison = "below"
        elif current_price > high:
            comparison = "above"
        else:
            comparison = "within"
        summary += f" Current price is {comparison} this scenario range."

    return summary + " This is for educational and research purposes only, not financial advice."


def _valuation_row(
    assumption: ScenarioAssumption,
    method: str,
    target_price: float | None,
    required_fields: tuple[str, ...],
) -> dict[str, Any]:
    missing_fields = [
        field_name for field_name in required_fields if _invalid_number(getattr(assumption, field_name))
    ]
    return {
        "scenario": assumption.scenario,
        "method": method,
        "assumed_inputs": asdict(assumption),
        "target_price": target_price if not missing_fields else None,
        "status": "insufficient_data" if missing_fields else "ok",
    }


def _ordered_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["scenario"] = pd.Categorical(frame["scenario"], categories=SCENARIO_ORDER, ordered=True)
    frame = frame.sort_values("scenario").reset_index(drop=True)
    frame["scenario"] = frame["scenario"].astype(str)
    return frame


def _validate_scenarios(assumptions: list[ScenarioAssumption]) -> None:
    scenarios = [assumption.scenario for assumption in assumptions]
    if scenarios != list(SCENARIO_ORDER):
        raise ValueError("Scenarios must be provided in Bear, Base, Bull order.")


def _scale(value: float | None, factor: float) -> float | None:
    if value is None:
        return None
    return float(value * factor)


def _multiply(left: float | None, right: float | None) -> float | None:
    if _invalid_number(left) or _invalid_number(right):
        return None
    return float(left * right)


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if _invalid_number(numerator) or _invalid_number(denominator) or denominator == 0:
        return None
    return float(numerator / denominator)


def _invalid_number(value: float | None) -> bool:
    return value is None or pd.isna(value) or value < 0


def with_updated_assumption(
    assumption: ScenarioAssumption,
    **changes: float | str | None,
) -> ScenarioAssumption:
    """Return a copy of a scenario assumption with selected fields changed."""

    return replace(assumption, **changes)
