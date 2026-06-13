# Stock Research Copilot

Stock Research Copilot is a personal finance research dashboard for learning, comparing stocks, and testing simple portfolio ideas.

It helps you:

- research an individual stock
- compare valuation scenarios
- generate a Markdown research report
- backtest a portfolio of stocks, ETFs, gold proxies, and fixed-interest investments
- upload or review source material when source research is enabled

This app is for education and research only. It is not financial advice.

## Start The App

From the project folder, run:

```bash
./run_app.sh
```

If that does not work, use:

```bash
UV_CACHE_DIR=.uv-cache uv run streamlit run src/app/streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Main Tools

Use the sidebar `Tool` selector to choose:

- `Stock Research`
- `Portfolio Backtest`

## Stock Research

Use this page when you want to research one company.

### Inputs

- `Ticker`: stock symbol such as `AAPL`, `NVDA`, `MSFT`, or `TSLA`
- `Period`: chart and metric history, such as `1Y`, `3Y`, `5Y`, `10Y`, or `Max`
- `Peers`: comparable tickers for the peer table
- `Valuation Method`: choose `blended`, `blended + DCF`, `PE`, `PS`, or `DCF`
- `Scenario Assumptions`: adjust EPS, P/E, revenue, P/S, free cash flow, DCF growth, discount rate, terminal growth, and net debt

### What You See

- price trend chart with moving averages
- key metrics such as close price, RSI, volatility, revenue growth, margins, and free cash flow margin
- financial performance charts and tables when data is available
- valuation scenarios for Bear/Base/Bull cases
- market-implied expectations
- peer comparison
- research report preview, download, and save options

### Valuation Scenarios

Valuation scenarios answer:

```text
What could the stock be worth under my Bear/Base/Bull assumptions?
```

These are model outputs from your assumptions. They are not predictions.

### Market-Implied Expectations

Market-implied expectations answer:

```text
What growth rate does the current stock price seem to imply?
```

This reverse-solves a simplified DCF model. It is a way to compare today’s market price with your assumptions.

### Research Report

Click `Generate Report Preview` to create a Markdown report inside the dashboard.

You can then:

- preview it
- download it
- save it under `data/reports/`

## Portfolio Backtest

Use this page when you want to estimate how a portfolio would have performed historically.

It supports:

- stocks
- ETFs
- bond ETFs
- gold/silver ETF proxies
- custom tickers
- custom fixed-interest investments
- one-time investments
- daily, monthly, or yearly recurring investments

### Basic Use

1. Choose `Portfolio Backtest` in the sidebar.
2. Set `Initial Investment`.
3. Choose `Backtest Period`.
4. Optionally set `Recurring Investment`.
5. Choose `Recurring Frequency`.
6. Add asset rows.
7. Make sure percentages add up to `100%`.
8. Click `Calculate Portfolio`.

### Asset Rows

Each row has:

- `Asset`
- optional custom ticker or interest rate field
- `Percent`
- `Remove`

Use `Add Asset` to add more rows.

Use `Auto Fill %` to split `100%` evenly across the current rows.

### Example Portfolio

| Asset | Percent |
| --- | ---: |
| SPY | 50 |
| AAPL | 30 |
| GLD | 20 |

The total must be exactly `100%` before calculation.

### Other Ticker

Choose `Other ticker...` when the asset is not in the dropdown.

Examples:

- `AMD`
- `COST`
- `SCHD`
- `ARKK`
- `BTC-USD`
- `0700.HK`
- `VOD.L`

The app will try to fetch the ticker with Yahoo Finance data.

### Custom Fixed Interest

Choose:

```text
CUSTOM_INTEREST - Custom Fixed Interest Investment
```

Use this for something like a savings product, fixed-rate account, or other custom return assumption.

Enter the annual interest rate as a decimal:

| Annual Interest | Meaning |
| ---: | --- |
| `0.04` | 4% per year |
| `0.025` | 2.5% per year |
| `0.00` | 0% per year |

The app creates a synthetic fixed-return path and mixes it with the rest of your portfolio.

### Portfolio Results

The backtest shows:

- total contributed
- final value
- profit/loss
- total return
- annualized return
- annualized volatility
- max drawdown
- portfolio value chart
- per-asset contribution table

For recurring investments, profit/loss and total return are calculated against total contributed capital.

## Optional API Keys

The app works without paid APIs for the main stock and portfolio tools.

Some optional source-research features use `.env` keys:

```bash
ALPHA_VANTAGE_API_KEY=
FMP_API_KEY=
FINNHUB_API_KEY=
SEC_USER_AGENT=StockResearchCopilot/0.1 your.email@example.com
```

Use a descriptive `SEC_USER_AGENT` if you download SEC filings.

## Data Sources

The app may use:

- Yahoo Finance through `yfinance`
- SEC EDGAR for filings
- Alpha Vantage for optional news or transcript features
- local uploaded files for transcripts or research reports

Market and financial data can be delayed, incomplete, missing, or temporarily unavailable.

## Common Issues

### The App Cannot Find A Ticker

Try:

- checking the ticker spelling
- using the exchange suffix, such as `.L`, `.HK`, or `.TO`
- refreshing later if Yahoo Finance is temporarily unavailable

### Portfolio Percentages Do Not Calculate

The asset percentages must add up to `100%`.

Use `Auto Fill %` if you want the app to split the allocation evenly.

### Streamlit Shows An Old Error

Refresh the browser page. If the error remains, stop and restart the app:

```bash
./run_app.sh
```

## Disclaimer

This project is for educational and research purposes only. It does not provide financial advice, investment recommendations, or trading signals. The analysis is based on public data and user assumptions, which may be incomplete or inaccurate.
