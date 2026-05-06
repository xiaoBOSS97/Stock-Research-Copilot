# Stock Research Copilot

A Python and Streamlit based equity research dashboard for educational use.

Stock Research Copilot collects public market and financial data, computes technical indicators and financial ratios, builds scenario-based valuation ranges, and generates structured Markdown equity research reports.

## Project Status

This repository is currently at the initial scaffold stage. The implemented v0.1 foundation includes:

- repository structure from `docs/stock_research_copilot_codex_spec.pdf`
- `yfinance` price history and company profile loader
- moving averages, RSI, annualized volatility, and max drawdown calculations
- focused unit tests for technical analysis behavior

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
