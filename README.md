# Stock Research Copilot

A Python and Streamlit based equity research dashboard for educational use.

Stock Research Copilot collects public market and financial data, computes technical indicators and financial ratios, builds scenario-based valuation ranges, and generates structured Markdown equity research reports.

## Project Status

This repository is at a stable MVP stage. The implemented v0.1 foundation includes:

- repository structure from `docs/stock_research_copilot_codex_spec.pdf`
- `yfinance` price history and company profile loader with ticker validation and optional raw CSV caching
- moving averages, RSI, annualized volatility, and max drawdown calculations
- yfinance financial statement loading plus revenue growth, margins, returns, leverage, and free cash flow metrics
- Bear/Base/Bull valuation scenarios using P/E, P/S, DCF, and blended methods
- Markdown equity research report generation
- Streamlit dashboard for price trends, financial metrics, valuation scenarios, peers, and report preview
- local earnings call transcript upload, theme analysis, and transcript passage search
- SEC EDGAR ticker-to-CIK lookup, recent filing metadata retrieval, and filing download helpers
- SEC filing HTML/text parsing into Business, Risk Factors, MD&A, and other source sections
- local filing RAG-style Q&A with cited source passages and safe refusal when no source is found
- optional SEC Filing Summary section in generated reports when parsed filings are available
- recent company news loading from Alpha Vantage News Sentiment with local JSON caching
- local financial organization report upload, extraction, topic analysis, and search
- market-implied expectation analysis for reverse-solved DCF growth and scenario deviation
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
- `DcfAssumption`
- `build_scenarios(...)`
- `build_dcf_scenarios(...)`
- `estimate_by_pe(assumptions)`
- `estimate_by_ps(assumptions)`
- `estimate_by_dcf(assumptions)`
- `blended_valuation(assumptions)`
- `blended_valuation_with_dcf(assumptions, dcf_assumptions)`
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
- Bear/Base/Bull valuation table with P/E, P/S, DCF, or blended methods
- peer comparison table with technical metrics, market cap, revenue growth, and net margin where available
- earnings call transcript analysis for uploaded `.txt` or `.md` transcripts
- recent news headlines, summaries, source links, and sentiment labels when `ALPHA_VANTAGE_API_KEY` is set
- financial organization report upload and local search for reports you have permission to use
- Markdown report preview, download, and save action

Run it with:

```bash
uv run streamlit run src/app/streamlit_app.py
```

## Earnings Call Transcript Analysis

The transcript analysis module lives in `src/rag/transcript_analysis.py` and currently supports:

- uploading or saving plain-text / Markdown earnings call transcripts under `data/transcripts/{TICKER}/`
- automatic transcript download from Alpha Vantage when `ALPHA_VANTAGE_API_KEY` is set
- deterministic transcript chunking and keyword retrieval
- topic signals for revenue/demand, margins/costs, guidance/outlook, risks/headwinds, and cash flow/capital
- management tone classification
- transcript question search that returns source passages

To enable automatic downloads, add an API key to `.env`:

```bash
ALPHA_VANTAGE_API_KEY=your_key_here
```

The automatic downloader uses Alpha Vantage's `EARNINGS_CALL_TRANSCRIPT` endpoint with quarters like `2024Q1`. Manual upload remains available when no API key is configured.

This is a local, lightweight RAG-style foundation. It does not require an LLM or embedding database yet.

## Recent News

The recent news loader lives in `src/data_loader/news_loader.py` and currently supports:

- `fetch_alpha_vantage_news(ticker)` using Alpha Vantage's `NEWS_SENTIMENT` endpoint
- normalized article rows with published date, title, source, summary, URL, and sentiment fields
- optional JSON caching under `data/news/{TICKER}_latest_news.json`
- dashboard display in the `Source Research > News` tab
- optional `Recent News` section in generated reports when cached articles are available

To enable live news downloads, add an API key to `.env`:

```bash
ALPHA_VANTAGE_API_KEY=your_key_here
```

If the live request fails, the dashboard falls back to cached news for the ticker when available.

## Financial Organization Reports

The research report loader lives in `src/data_loader/research_report_loader.py`, and the local report analysis helpers live in `src/rag/research_report_qa.py`.

This feature supports:

- uploading `.pdf`, `.txt`, or `.md` reports you have permission to use
- extracting report text locally and saving it under `data/research_reports/{TICKER}/`
- topic signals for rating/recommendation, price target/valuation, growth drivers, margins, and risks
- keyword search over uploaded report passages
- optional `External Research Notes` in generated reports when a local report is available

PDF extraction uses `pypdf`; run `uv sync` after pulling this feature so the dependency is installed.

## SEC Filing Loader

The SEC filing loader lives in `src/data_loader/sec_loader.py` and currently supports:

- `lookup_cik(ticker)` using the SEC ticker mapping
- `get_recent_filings(ticker, forms=("10-K", "10-Q", "8-K"), limit=10)`
- `get_latest_filing(ticker, form="10-K")`
- `download_filing(filing)` and `download_latest_filing(ticker, form="10-K")`

SEC filings are saved under `data/filings/{TICKER}/` when downloaded. Set a descriptive SEC user agent in `.env`:

```bash
SEC_USER_AGENT=StockResearchCopilot/0.1 your.email@example.com
```

## Filing Parser

The filing parser lives in `src/preprocessing/filing_parser.py` and currently supports:

- `load_filing_text(path)` for saved `.htm`, `.html`, or `.txt` filings
- `extract_filing_sections(text)` for major SEC items such as Business, Risk Factors, MD&A, and Financial Statements
- `chunk_filing_sections(sections, ticker=..., filing_type=..., filing_date=...)`
- `save_parsed_filing(sections, ticker=..., accession_number=...)`

Parsed filing section JSON is saved under `data/processed/filings/{TICKER}/`. This is the foundation for filing-based RAG Q&A and source-backed report sections.

## Filing Q&A

The filing Q&A module lives in `src/rag/filing_qa.py` and currently supports:

- listing and loading parsed filing section JSON files
- keyword retrieval over parsed filing chunks
- answers that cite source passages like `[1]`
- safe refusal when no relevant filing source is found

The Streamlit dashboard includes an SEC Filing Q&A section where you can download and parse the latest 10-K, 10-Q, or 8-K, then ask questions against the parsed filing.

Generated Markdown reports automatically include an **SEC Filing Summary** section when a parsed filing exists for the ticker.

## Market-Implied Expectations

The market-implied expectations module lives in `src/analysis/market_implied_expectations.py` and currently supports:

- reverse-solving the annual free cash flow growth rate implied by the current share price in the simplified DCF model
- comparing current price against Bear/Base/Bull valuation scenarios
- dashboard and report summaries that separate model assumptions from market-implied outputs

This is not a prediction. It answers: "what growth rate would make the current price approximately fit these DCF assumptions?"

## Data Sources

The MVP is designed to run without paid APIs. It uses public data sources such as:

- Yahoo Finance data through `yfinance`
- SEC EDGAR APIs for future official filing and company facts support
- SEC EDGAR filing metadata and filing primary documents

Network data may be incomplete, delayed, unavailable, or shaped differently across companies.

## MVP Limitations

- yfinance may return empty financial statements or temporary network errors.
- Price data is cached to `data/raw` and financial statements are cached to `data/processed` when available.
- P/E, P/S, and DCF valuation assumptions are user-controlled scenario inputs, not predictions.
- Peer comparison uses available public data and may show `N/A` when financial data is missing.
- embedding-based filing RAG, richer peer selection, and PDF export are planned future extensions.

## Disclaimer

This project is for educational and research purposes only. It does not provide financial advice, investment recommendations, or trading signals. The analysis is based on public data and model assumptions, which may be incomplete or inaccurate.
