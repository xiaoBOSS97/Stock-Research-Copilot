"""Buy-and-hold portfolio backtesting helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


TRADING_DAYS_PER_YEAR = 252
RECURRING_FREQUENCIES = {"none", "daily", "monthly", "yearly"}


class PortfolioBacktestError(ValueError):
    """Raised when a portfolio backtest cannot be calculated."""


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Portfolio backtest output used by dashboard and tests."""

    initial_investment: float
    total_contributed: float
    recurring_contribution: float
    contribution_frequency: str
    final_value: float
    profit_loss: float
    total_return: float
    annualized_return: float | None
    annualized_volatility: float | None
    max_drawdown: float | None
    start_date: str
    end_date: str
    trading_days: int
    portfolio_value: pd.Series
    daily_returns: pd.Series
    contributions: pd.Series
    asset_contributions: pd.DataFrame

    def metrics_dict(self) -> dict[str, Any]:
        """Return scalar metrics without the time series payload."""

        data = asdict(self)
        data.pop("portfolio_value")
        data.pop("daily_returns")
        data.pop("contributions")
        data.pop("asset_contributions")
        return data


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize positive portfolio weights to sum to one."""

    if not weights:
        raise PortfolioBacktestError("At least one portfolio asset is required.")

    normalized_symbols: dict[str, float] = {}
    for ticker, weight in weights.items():
        symbol = str(ticker).strip().upper()
        if not symbol:
            raise PortfolioBacktestError("Portfolio tickers must not be empty.")
        numeric_weight = pd.to_numeric(weight, errors="coerce")
        if pd.isna(numeric_weight):
            raise PortfolioBacktestError(f"Weight for {symbol} must be numeric.")
        numeric_weight = float(numeric_weight)
        if numeric_weight < 0:
            raise PortfolioBacktestError(f"Weight for {symbol} must not be negative.")
        normalized_symbols[symbol] = normalized_symbols.get(symbol, 0.0) + numeric_weight

    total_weight = sum(normalized_symbols.values())
    if total_weight <= 0:
        raise PortfolioBacktestError("At least one portfolio weight must be positive.")

    return {ticker: weight / total_weight for ticker, weight in normalized_symbols.items()}


def close_prices_from_history(price_history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an aligned close-price table from ticker price history frames."""

    if not price_history:
        raise PortfolioBacktestError("No price history supplied.")

    series = {}
    for ticker, history in price_history.items():
        symbol = str(ticker).strip().upper()
        if history.empty or "Close" not in history.columns:
            raise PortfolioBacktestError(f"Price history for {symbol} is missing Close prices.")
        close = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if close.empty:
            raise PortfolioBacktestError(f"Price history for {symbol} has no usable Close prices.")
        close.index = pd.to_datetime(close.index)
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
        series[symbol] = close.sort_index()

    prices = pd.DataFrame(series).dropna(how="any")
    if len(prices) < 2:
        raise PortfolioBacktestError("Portfolio needs at least two overlapping price rows.")
    return prices


def calculate_buy_and_hold_values(
    close_prices: pd.DataFrame,
    weights: dict[str, float],
    initial_investment: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate buy-and-hold portfolio value and per-asset value history."""

    portfolio_value, asset_values, _, _ = calculate_portfolio_values(
        close_prices,
        weights,
        initial_investment,
        recurring_contribution=0.0,
        contribution_frequency="none",
    )
    return portfolio_value, asset_values


def calculate_portfolio_values(
    close_prices: pd.DataFrame,
    weights: dict[str, float],
    initial_investment: float,
    recurring_contribution: float = 0.0,
    contribution_frequency: str = "none",
) -> tuple[pd.Series, pd.DataFrame, pd.Series, dict[str, float]]:
    """Calculate portfolio value with optional recurring contributions."""

    if initial_investment < 0:
        raise PortfolioBacktestError("Initial investment must not be negative.")
    if recurring_contribution < 0:
        raise PortfolioBacktestError("Recurring contribution must not be negative.")
    frequency = normalize_contribution_frequency(contribution_frequency)
    if initial_investment <= 0 and recurring_contribution <= 0:
        raise PortfolioBacktestError("Initial investment or recurring contribution must be positive.")

    normalized_weights = normalize_weights(weights)
    missing = [ticker for ticker in normalized_weights if ticker not in close_prices.columns]
    if missing:
        raise PortfolioBacktestError(f"Missing price data for portfolio assets: {missing}.")

    prices = close_prices.loc[:, list(normalized_weights)].dropna(how="any")
    if len(prices) < 2:
        raise PortfolioBacktestError("Portfolio needs at least two overlapping price rows.")

    shares = pd.Series(0.0, index=prices.columns)
    invested_by_asset = {ticker: 0.0 for ticker in prices.columns}
    contribution_dates = contribution_schedule(prices.index, frequency)
    contribution_rows = []
    asset_value_rows = []

    first_date = prices.index[0]
    if initial_investment > 0:
        first_prices = prices.loc[first_date]
        for ticker, weight in normalized_weights.items():
            contribution = initial_investment * weight
            shares[ticker] += contribution / first_prices[ticker]
            invested_by_asset[ticker] += contribution

    for date, price_row in prices.iterrows():
        contribution = 0.0
        if date in contribution_dates and recurring_contribution > 0:
            contribution = recurring_contribution
            for ticker, weight in normalized_weights.items():
                asset_contribution = recurring_contribution * weight
                shares[ticker] += asset_contribution / price_row[ticker]
                invested_by_asset[ticker] += asset_contribution
        contribution_rows.append(contribution)
        asset_value_rows.append((price_row * shares).to_dict())

    asset_values = pd.DataFrame(asset_value_rows, index=prices.index, columns=prices.columns)
    portfolio_value = asset_values.sum(axis=1).rename("Portfolio Value")
    contributions = pd.Series(contribution_rows, index=prices.index, name="Contribution")
    if initial_investment > 0:
        contributions.iloc[0] += initial_investment
    return portfolio_value, asset_values, contributions, invested_by_asset


def normalize_contribution_frequency(frequency: str) -> str:
    """Normalize recurring contribution frequency labels."""

    normalized = str(frequency or "none").strip().lower()
    if normalized not in RECURRING_FREQUENCIES:
        raise PortfolioBacktestError("Contribution frequency must be none, daily, monthly, or yearly.")
    return normalized


def contribution_schedule(index: pd.DatetimeIndex, frequency: str) -> set[pd.Timestamp]:
    """Return dates when recurring contributions should be invested."""

    normalized = normalize_contribution_frequency(frequency)
    dates = pd.DatetimeIndex(index).sort_values()
    if len(dates) < 2 or normalized == "none":
        return set()
    if normalized == "daily":
        return set(dates[1:])

    frame = pd.DataFrame({"date": dates})
    if normalized == "monthly":
        grouped = frame.groupby([frame["date"].dt.year, frame["date"].dt.month], sort=True)
    else:
        grouped = frame.groupby(frame["date"].dt.year, sort=True)

    scheduled = [group["date"].iloc[0] for _, group in grouped]
    return {date for date in scheduled if date != dates[0]}


def calculate_max_drawdown(portfolio_value: pd.Series) -> float | None:
    """Return maximum peak-to-trough drawdown for a portfolio value series."""

    values = pd.to_numeric(portfolio_value, errors="coerce").dropna()
    if values.empty:
        return None
    running_peak = values.cummax()
    drawdown = values / running_peak - 1
    return float(drawdown.min())


def calculate_annualized_return(portfolio_value: pd.Series) -> float | None:
    """Return CAGR for the portfolio value series."""

    values = pd.to_numeric(portfolio_value, errors="coerce").dropna()
    if len(values) < 2 or values.iloc[0] <= 0:
        return None
    elapsed_days = max((values.index[-1] - values.index[0]).days, 1)
    years = elapsed_days / 365.25
    if years <= 0:
        return None
    return float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)


def calculate_annualized_volatility(daily_returns: pd.Series) -> float | None:
    """Return annualized volatility from daily portfolio returns."""

    returns = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(returns) < 2:
        return None
    return float(returns.std() * (TRADING_DAYS_PER_YEAR**0.5))


def calculate_time_weighted_annualized_return(daily_returns: pd.Series) -> float | None:
    """Annualize compounded daily returns after removing external cash flows."""

    returns = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if returns.empty:
        return None
    total_growth = float((1 + returns).prod())
    years = len(returns) / TRADING_DAYS_PER_YEAR
    if years <= 0 or total_growth <= 0:
        return None
    return total_growth ** (1 / years) - 1


def calculate_cash_adjusted_returns(
    portfolio_value: pd.Series,
    contributions: pd.Series,
) -> pd.Series:
    """Return daily returns adjusted for external contributions."""

    values = pd.to_numeric(portfolio_value, errors="coerce")
    flows = pd.to_numeric(contributions, errors="coerce").fillna(0.0)
    returns = []
    index = []
    for position in range(1, len(values)):
        previous_value = values.iloc[position - 1]
        if pd.isna(previous_value) or previous_value <= 0:
            continue
        current_value = values.iloc[position]
        flow = flows.iloc[position]
        if pd.isna(current_value):
            continue
        returns.append((current_value - previous_value - flow) / previous_value)
        index.append(values.index[position])
    return pd.Series(returns, index=index, name="Daily Return")


def asset_contribution_frame(
    asset_values: pd.DataFrame,
    weights: dict[str, float],
    invested_by_asset: dict[str, float] | float,
) -> pd.DataFrame:
    """Build a per-asset contribution table for a buy-and-hold portfolio."""

    normalized_weights = normalize_weights(weights)
    if isinstance(invested_by_asset, dict):
        contributed_by_asset = invested_by_asset
    else:
        contributed_by_asset = {
            ticker: float(invested_by_asset) * normalized_weights[ticker]
            for ticker in asset_values.columns
        }
    rows = []
    final_total = float(asset_values.iloc[-1].sum())
    for ticker in asset_values.columns:
        initial_value = contributed_by_asset[ticker]
        final_value = float(asset_values[ticker].iloc[-1])
        profit_loss = final_value - initial_value
        rows.append(
            {
                "ticker": ticker,
                "weight": normalized_weights[ticker],
                "initial_value": initial_value,
                "final_value": final_value,
                "profit_loss": profit_loss,
                "total_return": (final_value / initial_value - 1) if initial_value > 0 else None,
                "ending_weight": final_value / final_total if final_total > 0 else None,
            }
        )
    return pd.DataFrame(rows)


def run_buy_and_hold_backtest(
    price_history: dict[str, pd.DataFrame],
    weights: dict[str, float],
    initial_investment: float,
    recurring_contribution: float = 0.0,
    contribution_frequency: str = "none",
) -> PortfolioBacktestResult:
    """Run a buy-and-hold portfolio backtest from historical price frames."""

    close_prices = close_prices_from_history(price_history)
    portfolio_value, asset_values, contributions, invested_by_asset = calculate_portfolio_values(
        close_prices,
        weights,
        initial_investment,
        recurring_contribution=recurring_contribution,
        contribution_frequency=contribution_frequency,
    )
    frequency = normalize_contribution_frequency(contribution_frequency)
    daily_returns = calculate_cash_adjusted_returns(portfolio_value, contributions)
    if daily_returns.empty:
        daily_returns = portfolio_value.pct_change().dropna().rename("Daily Return")
    final_value = float(portfolio_value.iloc[-1])
    total_contributed = float(contributions.sum())
    profit_loss = final_value - total_contributed
    total_return = final_value / total_contributed - 1

    return PortfolioBacktestResult(
        initial_investment=float(initial_investment),
        total_contributed=total_contributed,
        recurring_contribution=float(recurring_contribution),
        contribution_frequency=frequency,
        final_value=final_value,
        profit_loss=profit_loss,
        total_return=float(total_return),
        annualized_return=calculate_time_weighted_annualized_return(daily_returns),
        annualized_volatility=calculate_annualized_volatility(daily_returns),
        max_drawdown=calculate_max_drawdown(portfolio_value),
        start_date=portfolio_value.index[0].date().isoformat(),
        end_date=portfolio_value.index[-1].date().isoformat(),
        trading_days=len(portfolio_value),
        portfolio_value=portfolio_value,
        daily_returns=daily_returns,
        contributions=contributions,
        asset_contributions=asset_contribution_frame(asset_values, weights, invested_by_asset),
    )
