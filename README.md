# Stock Research Copilot

A Python and Streamlit based equity research dashboard for educational use.

Stock Research Copilot collects public market and financial data, computes technical indicators and financial ratios, builds scenario-based valuation ranges, and generates structured Markdown equity research reports.

## Project Status

This repository is currently at the initial scaffold stage. The implemented v0.1 foundation includes:

- repository structure from `docs/stock_research_copilot_codex_spec.pdf`
- `yfinance` price history and company profile loader with ticker validation and optional raw CSV caching
- moving averages, RSI, annualized volatility, and max drawdown calculations
- yfinance financial statement loading plus revenue growth, margins, returns, leverage, and free cash flow metrics
- focused unit tests for technical analysis behavior

## Price Loader

The price loader lives in `src/data_loader/price_loader.py` and currently supports:

- `get_price_history(ticker, period="5y", interval="1d")`
- `get_company_profile(ticker)`
- `validate_ticker(ticker)`
- optional CSV caching to `data/raw`

## Technical Analysis

The technical analysis module lives in `src/analysis/technical_analysis.py` and currently supports:

- `add_moving_averages(price_data)` for MA20, MA50, and MA200
- `calculate_rsi(price_data, window=14)`
- `calculate_volatility(price_data)`
- `calculate_max_drawdown(price_data)`
- `add_technical_indicators(price_data)`
- `calculate_technical_metrics(price_data)` for dashboard/report summaries
- `generate_technical_summary(price_data)` with disclaimer-safe wording

## Financial Analysis

The financial loader lives in `src/data_loader/financial_loader.py` and currently supports:

- `get_financial_statements(ticker)`
- `normalize_financials(raw)`
- `get_ttm_metrics(ticker)`

The financial ratios module lives in `src/analysis/financial_ratios.py` and currently supports:

- `calculate_growth_rates(income_statement)`
- `calculate_margins(income_statement, cash_flow)`
- `calculate_free_cash_flow(cash_flow)`
- `calculate_returns(income_statement, balance_sheet)`
- `calculate_leverage(income_statement, balance_sheet)`
- `calculate_financial_metrics(statements)`
- `generate_financial_summary(metrics)`

## Quick Start

```bash
uv sync
uv run pytest -q
```

Run the dashboard once the Streamlit app is implemented:

```bash
uv run streamlit run src/app/streamlit_app.py
```

If you prefer an activated shell, `uv sync` creates `.venv`, so you can still run:

```bash
source .venv/bin/activate
pytest -q
streamlit run src/app/streamlit_app.py
```

## Data Sources

The MVP is designed to run without paid APIs. It uses public data sources such as:

- Yahoo Finance data through `yfinance`
- SEC EDGAR APIs for future official filing and company facts support

Network data may be incomplete, delayed, unavailable, or shaped differently across companies.

## Disclaimer

This project is for educational and research purposes only. It does not provide financial advice, investment recommendations, or trading signals. The analysis is based on public data and model assumptions, which may be incomplete or inaccurate.
