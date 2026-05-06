# Stock Research Copilot

A Python and Streamlit based equity research dashboard for educational use.

Stock Research Copilot collects public market and financial data, computes technical indicators and financial ratios, builds scenario-based valuation ranges, and generates structured Markdown equity research reports.

## Project Status

This repository is at a stable MVP stage. The implemented v0.1 foundation includes:

- repository structure from `docs/stock_research_copilot_codex_spec.pdf`
- `yfinance` price history and company profile loader with ticker validation and optional raw CSV caching
- moving averages, RSI, annualized volatility, and max drawdown calculations
- yfinance financial statement loading plus revenue growth, margins, returns, leverage, and free cash flow metrics
- Bear/Base/Bull valuation scenarios using PE, P/S, and blended methods
- Markdown equity research report generation
- Streamlit dashboard for price trends, financial metrics, valuation scenarios, peers, and report preview
- unit tests for loaders, analysis modules, valuation, reports, formatting, and dashboard helpers

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

## Valuation

The valuation module lives in `src/analysis/valuation.py` and currently supports:

- `ScenarioAssumption`
- `build_scenarios(...)`
- `estimate_by_pe(assumptions)`
- `estimate_by_ps(assumptions)`
- `blended_valuation(assumptions)`
- `summarize_valuation(valuation_table, current_price=None)`

Valuation outputs are scenario ranges based on explicit assumptions. They are for educational and research purposes only and are not investment advice.

## Report Generation

The report generator lives in `src/report/report_generator.py` and currently supports:

- `build_report_context(...)`
- `render_markdown_report(context)`
- `save_report(markdown, ticker)`
- `generate_report_for_ticker(ticker)`

Generate a Markdown report:

```bash
uv run python -m src.report.report_generator --ticker AAPL
```

The default output path is `data/reports/AAPL_report.md`.

## Quick Start

```bash
uv sync
uv run pytest -q
```

Run the dashboard:

```bash
uv run streamlit run src/app/streamlit_app.py
```

Generate a report:

```bash
uv run python -m src.report.report_generator --ticker AAPL
```

If you prefer an activated shell, `uv sync` creates `.venv`, so you can still run:

```bash
source .venv/bin/activate
pytest -q
streamlit run src/app/streamlit_app.py
```

## Dashboard

The Streamlit dashboard lives in `src/app/streamlit_app.py` and includes:

- ticker, period, peer, valuation method, and scenario assumption controls
- close price with MA20, MA50, and MA200
- technical and financial metric cards
- revenue, net income, and free cash flow chart when statements are available
- Bear/Base/Bull valuation table
- peer comparison table with technical metrics, market cap, revenue growth, and net margin where available
- Markdown report preview, download, and save action

Run it with:

```bash
uv run streamlit run src/app/streamlit_app.py
```

## Data Sources

The MVP is designed to run without paid APIs. It uses public data sources such as:

- Yahoo Finance data through `yfinance`
- SEC EDGAR APIs for future official filing and company facts support

Network data may be incomplete, delayed, unavailable, or shaped differently across companies.

## MVP Limitations

- yfinance may return empty financial statements or temporary network errors.
- Price data is cached to `data/raw` and financial statements are cached to `data/processed` when available.
- P/E and P/S valuation assumptions are user-controlled scenario inputs, not predictions.
- Peer comparison uses available public data and may show `N/A` when financial data is missing.
- SEC EDGAR support, DCF valuation, richer peer selection, and PDF export are planned future extensions.

## Disclaimer

This project is for educational and research purposes only. It does not provide financial advice, investment recommendations, or trading signals. The analysis is based on public data and model assumptions, which may be incomplete or inaccurate.
