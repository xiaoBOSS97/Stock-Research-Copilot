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


@dataclass(frozen=True)
class DcfAssumption:
    """Inputs used by a simple free-cash-flow DCF scenario."""

    scenario: str
    base_free_cash_flow: float | None
    growth_rate: float | None
    discount_rate: float | None
    terminal_growth_rate: float | None
    net_debt: float | None
    shares_outstanding: float | None
    projection_years: int = 5


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


def build_dcf_scenarios(
    base_free_cash_flow: float | None,
    discount_rate: float,
    terminal_growth_rate: float,
    shares_outstanding: float | None,
    net_debt: float | None = 0.0,
    projection_years: int = 5,
    base_growth_rate: float = 0.05,
    custom_assumptions: Iterable[DcfAssumption] | None = None,
) -> list[DcfAssumption]:
    """Build Bear/Base/Bull DCF assumptions from base inputs."""

    if custom_assumptions is not None:
        assumptions = list(custom_assumptions)
        _validate_scenarios(assumptions)
        return assumptions

    assumptions = [
        DcfAssumption(
            scenario="Bear",
            base_free_cash_flow=base_free_cash_flow,
            growth_rate=base_growth_rate - 0.03,
            discount_rate=discount_rate + 0.02,
            terminal_growth_rate=max(terminal_growth_rate - 0.01, 0.0),
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
            projection_years=projection_years,
        ),
        DcfAssumption(
            scenario="Base",
            base_free_cash_flow=base_free_cash_flow,
            growth_rate=base_growth_rate,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
            projection_years=projection_years,
        ),
        DcfAssumption(
            scenario="Bull",
            base_free_cash_flow=base_free_cash_flow,
            growth_rate=base_growth_rate + 0.03,
            discount_rate=max(discount_rate - 0.01, 0.0),
            terminal_growth_rate=terminal_growth_rate + 0.005,
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
            projection_years=projection_years,
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


def estimate_by_dcf(assumptions: Iterable[DcfAssumption]) -> pd.DataFrame:
    """Estimate equity value per share using a simple FCF DCF model."""

    rows = []
    for assumption in assumptions:
        target_price, details = _dcf_target_price(assumption)
        rows.append(
            {
                "scenario": assumption.scenario,
                "method": "DCF",
                "assumed_inputs": asdict(assumption) | details,
                "target_price": target_price,
                "status": "ok" if target_price is not None else "insufficient_data",
            }
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


def blended_valuation_with_dcf(
    assumptions: Iterable[ScenarioAssumption],
    dcf_assumptions: Iterable[DcfAssumption] | None = None,
    pe_weight: float = 0.4,
    ps_weight: float = 0.3,
    dcf_weight: float = 0.3,
) -> pd.DataFrame:
    """Blend PE, P/S, and optional DCF valuation outputs."""

    if pe_weight < 0 or ps_weight < 0 or dcf_weight < 0:
        raise ValueError("Valuation weights must be non-negative.")
    total_weight = pe_weight + ps_weight + dcf_weight
    if total_weight == 0:
        raise ValueError("At least one valuation weight must be positive.")

    pe = estimate_by_pe(assumptions).set_index("scenario")
    ps = estimate_by_ps(assumptions).set_index("scenario")
    dcf = estimate_by_dcf(dcf_assumptions).set_index("scenario") if dcf_assumptions is not None else None

    rows = []
    for scenario in SCENARIO_ORDER:
        prices = {
            "pe_target_price": pe.loc[scenario, "target_price"] if scenario in pe.index else None,
            "ps_target_price": ps.loc[scenario, "target_price"] if scenario in ps.index else None,
            "dcf_target_price": dcf.loc[scenario, "target_price"] if dcf is not None and scenario in dcf.index else None,
        }
        weighted_prices = []
        for key, weight in [
            ("pe_target_price", pe_weight),
            ("ps_target_price", ps_weight),
            ("dcf_target_price", dcf_weight),
        ]:
            price = prices[key]
            if pd.notna(price):
                weighted_prices.append((float(price), weight))

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
                    "dcf_weight": dcf_weight / total_weight,
                    **{key: None if pd.isna(value) else float(value) for key, value in prices.items()},
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


def _dcf_target_price(assumption: DcfAssumption) -> tuple[float | None, dict[str, float | None]]:
    if (
        _invalid_number(assumption.base_free_cash_flow)
        or assumption.growth_rate is None
        or pd.isna(assumption.growth_rate)
        or assumption.growth_rate <= -1
        or _invalid_number(assumption.discount_rate)
        or _invalid_number(assumption.terminal_growth_rate)
        or _invalid_number(assumption.shares_outstanding)
    ):
        return None, {"enterprise_value": None, "equity_value": None, "terminal_value": None}
    if assumption.projection_years <= 0:
        return None, {"enterprise_value": None, "equity_value": None, "terminal_value": None}
    if assumption.discount_rate <= assumption.terminal_growth_rate:
        return None, {"enterprise_value": None, "equity_value": None, "terminal_value": None}
    if assumption.shares_outstanding == 0:
        return None, {"enterprise_value": None, "equity_value": None, "terminal_value": None}

    projected_fcf = []
    for year in range(1, assumption.projection_years + 1):
        projected_fcf.append(assumption.base_free_cash_flow * ((1 + assumption.growth_rate) ** year))

    present_value_fcf = sum(
        cash_flow / ((1 + assumption.discount_rate) ** year)
        for year, cash_flow in enumerate(projected_fcf, start=1)
    )
    final_fcf = projected_fcf[-1]
    terminal_value = final_fcf * (1 + assumption.terminal_growth_rate) / (
        assumption.discount_rate - assumption.terminal_growth_rate
    )
    present_value_terminal = terminal_value / ((1 + assumption.discount_rate) ** assumption.projection_years)
    enterprise_value = present_value_fcf + present_value_terminal
    equity_value = enterprise_value - (assumption.net_debt or 0.0)
    target_price = equity_value / assumption.shares_outstanding

    return float(target_price), {
        "enterprise_value": float(enterprise_value),
        "equity_value": float(equity_value),
        "terminal_value": float(terminal_value),
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
