# Stock Research Copilot Advanced

An AI-assisted equity research system for public companies.

This project combines market data ingestion, financial statement analysis, SEC filing retrieval, retrieval-augmented generation (RAG), valuation modeling, peer comparison, news/event analysis, watchlist monitoring, and automated equity research report generation.

> **Disclaimer**
> This project is for educational and research purposes only. It does not provide financial advice, investment recommendations, trading signals, or automated investment decisions.

---

## 1. Project Overview

**Stock Research Copilot Advanced** is a long-term personal project designed to behave like a lightweight AI equity research analyst.

Given a stock ticker, company name, or industry topic, the system should collect structured and unstructured data, analyze company fundamentals, compare the company with peers, evaluate valuation scenarios, and generate a source-grounded research report.

Example user input:

```text
NVDA
```

Expected workflow:

```text
1. Identify the company and its metadata.
2. Fetch historical stock prices and market metrics.
3. Fetch financial statements and key valuation metrics.
4. Download and parse SEC filings such as 10-K, 10-Q, and 8-K.
5. Extract relevant filing sections such as Business, Risk Factors, and MD&A.
6. Retrieve company news and analyst-related data where available.
7. Identify peer companies.
8. Calculate financial ratios and technical indicators.
9. Run PE, PS, EV/EBITDA, and DCF valuation models.
10. Generate Bear/Base/Bull valuation scenarios.
11. Reverse-engineer market-implied expectations from the current stock price.
12. Generate a structured research report with assumptions and sources.
13. Generate watchlist alerts and monitoring checklists.
```

The goal is not to predict stock prices with certainty. The goal is to build a transparent, explainable, and reproducible equity research workflow.

---

## 2. Core Product Goals

1. **Automated data collection**  
   Collect market prices, company profiles, financial statements, SEC filings, news, analyst data, and peer information.

2. **Structured fundamental analysis**  
   Calculate growth metrics, margins, returns, leverage, liquidity, cash flow quality, and valuation multiples.

3. **SEC filing understanding**  
   Parse annual and quarterly reports, build a RAG pipeline, and answer questions using source-grounded filing excerpts.

4. **Multi-model valuation**  
   Support relative valuation, DCF valuation, scenario analysis, sensitivity analysis, and market-implied expectation analysis.

5. **Peer and industry analysis**  
   Compare a target company against relevant peers across growth, profitability, cash flow, and valuation.

6. **News and event monitoring**  
   Summarize company news, classify events, estimate importance, and trigger alerts.

7. **Automated research reports**  
   Generate Markdown, HTML, and PDF research reports with assumptions, tables, charts, and source references.

8. **Watchlist and alerting**  
   Monitor selected companies for price moves, new filings, valuation deviations, rating changes, and important news events.

---

## 3. Scope

### 3.1 In Scope

- Historical stock price ingestion
- Company profile ingestion
- Financial statement ingestion
- SEC filing metadata retrieval
- SEC filing download and parsing
- Financial ratio calculation
- Technical indicator calculation
- PE, PS, EV/EBITDA, and DCF valuation
- Bear/Base/Bull scenario analysis
- Market-implied expectation analysis
- Peer comparison
- News/event ingestion and summarization
- Filing-based RAG question answering
- Research report generation
- Watchlist and alert engine
- Streamlit dashboard
- FastAPI backend
- Database persistence
- Unit tests and integration tests

### 3.2 Out of Scope

- Automated trading
- Brokerage integration
- Portfolio execution
- Personalized investment advice
- Guaranteed stock price prediction
- High-frequency trading strategies
- Scraping paid research reports
- Use of non-public information
- Direct buy/sell/hold recommendations

---

## 4. Recommended Tech Stack

| Layer | Recommended Tools |
|---|---|
| Programming language | Python |
| Data processing | pandas, numpy |
| Backend API | FastAPI |
| Frontend MVP | Streamlit |
| Advanced frontend | Next.js / React |
| Charts | Plotly, ECharts |
| Relational database | PostgreSQL |
| Time-series extension | TimescaleDB |
| Vector database | Chroma, Qdrant, or pgvector |
| Cache | Redis |
| Job scheduling | APScheduler, Celery, RQ, or Prefect |
| Filing parsing | BeautifulSoup, pdfplumber, unstructured |
| LLM integration | OpenAI API or local LLM |
| Embeddings | OpenAI embeddings or sentence-transformers |
| Testing | pytest |
| Logging | Python logging or loguru |
| Configuration | pydantic-settings, dotenv |
| Deployment | Docker, Docker Compose |

---

## 5. System Architecture

```text
                         User Interface
                    Streamlit / Next.js Frontend
                                  |
                                  v
                            FastAPI Backend
                                  |
        -----------------------------------------------------
        |                         |                         |
        v                         v                         v
 Data Ingestion Layer       Analysis Engine              AI Engine
        |                         |                         |
        v                         v                         v
 Market Data Loader        Financial Ratios            RAG Pipeline
 Financial Data Loader     Technical Indicators        Report Agent
 SEC Filing Loader         Valuation Models            Q&A Agent
 News Loader               Peer Comparison             Summary Agent
 Analyst Data Loader       Risk Scoring                Alert Agent
        |                         |                         |
        -----------------------------------------------------
                                  |
                                  v
                            Database Layer
        PostgreSQL + TimescaleDB + Vector DB + Object Storage
                                  |
                                  v
                    Reports / Alerts / Logs / Cache
```

---

## 6. Suggested Repository Structure

```text
stock-research-copilot-advanced/
|
├── README.md
├── CODEX_CONTEXT.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── Dockerfile
|
├── data/
│   ├── raw/
│   ├── processed/
│   ├── filings/
│   ├── vector_store/
│   └── reports/
|
├── notebooks/
│   ├── 01_market_data_exploration.ipynb
│   ├── 02_financial_ratios.ipynb
│   ├── 03_sec_filing_parsing.ipynb
│   ├── 04_rag_prototype.ipynb
│   ├── 05_valuation_model.ipynb
│   └── 06_report_generation.ipynb
|
├── src/
│   ├── config/
│   ├── database/
│   ├── data_ingestion/
│   ├── preprocessing/
│   ├── analysis/
│   ├── ai/
│   ├── reports/
│   ├── alerts/
│   ├── api/
│   └── app/
|
├── tests/
└── docs/
```

---

## 7. Core Modules

### 7.1 Data Ingestion

Responsible for retrieving and normalizing raw data.

Primary files:

- `src/data_ingestion/market_data_loader.py`
- `src/data_ingestion/financial_data_loader.py`
- `src/data_ingestion/sec_filing_loader.py`
- `src/data_ingestion/news_loader.py`
- `src/data_ingestion/analyst_loader.py`
- `src/data_ingestion/peer_loader.py`

Expected capabilities:

- Fetch historical OHLCV price data.
- Fetch company metadata.
- Fetch financial statements.
- Map ticker symbols to SEC CIK identifiers.
- Retrieve SEC filing lists.
- Download 10-K, 10-Q, and 8-K filings.
- Fetch recent news.
- Fetch analyst-related data where available.
- Cache data locally and avoid repeated API calls.

### 7.2 Preprocessing

Responsible for cleaning, validating, and transforming raw data.

Primary files:

- `src/preprocessing/data_cleaner.py`
- `src/preprocessing/filing_parser.py`
- `src/preprocessing/text_chunker.py`
- `src/preprocessing/quality_checker.py`

Expected capabilities:

- Normalize column names.
- Handle missing values.
- Validate time series continuity.
- Parse SEC filing HTML or text.
- Extract filing sections.
- Split filing text into source-traceable chunks.
- Attach metadata to each chunk.

### 7.3 Analysis Engine

Responsible for financial, technical, peer, risk, and valuation analysis.

Primary files:

- `src/analysis/financial_ratios.py`
- `src/analysis/technical_indicators.py`
- `src/analysis/valuation.py`
- `src/analysis/dcf_model.py`
- `src/analysis/peer_comparison.py`
- `src/analysis/market_implied_expectations.py`
- `src/analysis/risk_scoring.py`
- `src/analysis/news_sentiment.py`

Expected capabilities:

- Calculate financial ratios.
- Calculate moving averages, RSI, volatility, and drawdown.
- Run relative valuation.
- Run DCF valuation.
- Generate Bear/Base/Bull scenarios.
- Run sensitivity analysis.
- Compare company metrics with peers.
- Reverse-engineer market-implied growth or margin assumptions.

### 7.4 AI Engine

Responsible for embedding, retrieval, RAG question answering, summarization, and report writing.

Primary files:

- `src/ai/embeddings.py`
- `src/ai/vector_store.py`
- `src/ai/retriever.py`
- `src/ai/rag_qa.py`
- `src/ai/summarizer.py`
- `src/ai/report_agent.py`
- `src/ai/prompt_templates.py`

Expected capabilities:

- Generate embeddings for filing chunks.
- Store and retrieve chunks from a vector database.
- Answer filing-based questions with citations.
- Summarize filing sections.
- Extract risks, management discussion points, and business changes.
- Generate report sections based on structured data and retrieved text.

### 7.5 Reports

Responsible for rendering research reports.

Primary files:

- `src/reports/report_generator.py`
- `src/reports/markdown_renderer.py`
- `src/reports/pdf_exporter.py`
- `src/reports/templates/`

Expected capabilities:

- Generate Markdown reports.
- Generate HTML reports.
- Export PDF reports.
- Insert charts and tables.
- Include valuation assumptions.
- Include data sources and source references.

### 7.6 Alerts

Responsible for watchlist monitoring and rule-based alerts.

Primary files:

- `src/alerts/watchlist.py`
- `src/alerts/alert_rules.py`
- `src/alerts/alert_engine.py`
- `src/alerts/notification_service.py`

Expected capabilities:

- Add and remove tickers from a watchlist.
- Trigger alerts for price moves.
- Trigger alerts for new filings.
- Trigger alerts for valuation deviations.
- Trigger alerts for negative or important news.
- Generate daily watchlist summaries.

---

## 8. Database Schema Overview

The project should eventually support the following tables:

- `companies`
- `stock_prices`
- `financial_statements`
- `financial_ratios`
- `sec_filings`
- `filing_chunks`
- `news_events`
- `valuation_results`
- `watchlist`
- `alerts`

### 8.1 companies

```text
id
ticker
company_name
exchange
sector
industry
country
currency
cik
website
```

### 8.2 stock_prices

```text
id
ticker
date
open
high
low
close
adjusted_close
volume
```

### 8.3 financial_statements

```text
id
ticker
fiscal_year
fiscal_period
revenue
gross_profit
operating_income
net_income
eps
total_assets
total_liabilities
shareholders_equity
operating_cash_flow
capex
free_cash_flow
```

### 8.4 sec_filings

```text
id
ticker
cik
filing_type
filing_date
period_end_date
accession_number
file_url
local_path
parsed_status
```

### 8.5 filing_chunks

```text
id
filing_id
ticker
section
chunk_index
text
embedding_id
source_ref
```

### 8.6 valuation_results

```text
id
ticker
valuation_date
method
bear_value
base_value
bull_value
assumptions_json
notes
```

---

## 9. Valuation Methodology

### 9.1 PE Valuation

```text
Estimated Share Price = Expected EPS × PE Multiple
```

Use cases:

- Mature profitable companies
- Stable earnings companies
- Companies with meaningful EPS estimates

### 9.2 PS Valuation

```text
Estimated Share Price = Revenue Per Share × PS Multiple
```

Use cases:

- High-growth companies
- Companies with temporarily depressed earnings
- Software or platform businesses

### 9.3 EV/EBITDA Valuation

```text
Enterprise Value = EBITDA × EV/EBITDA Multiple
Equity Value = Enterprise Value - Net Debt
Estimated Share Price = Equity Value / Shares Outstanding
```

### 9.4 DCF Valuation

DCF should forecast free cash flow and discount it to present value.

Required assumptions:

```text
Revenue growth
Operating margin
Tax rate
Capital expenditure
Depreciation and amortization
Net working capital change
WACC
Terminal growth rate
Net debt or net cash
Shares outstanding
```

Core outputs:

```text
Projected revenue
Projected operating income
Projected free cash flow
Discounted cash flow value
Terminal value
Enterprise value
Equity value
Intrinsic value per share
```

### 9.5 Scenario Valuation

Each valuation method should support:

```text
Bear Case
Base Case
Bull Case
```

Each scenario must contain explicit assumptions.

### 9.6 Market-Implied Expectations

This module should answer:

```text
What future growth rate or margin level is already implied by the current stock price?
```

Expected outputs:

```text
Implied revenue CAGR
Implied operating margin
Required FCF growth
Current price deviation from Bear/Base/Bull scenarios
Conditions needed to justify the current valuation
```

---

## 10. RAG Pipeline

The filing-based RAG system should follow this workflow:

```text
SEC filing HTML/TXT/PDF
        |
        v
Text extraction
        |
        v
Section detection
        |
        v
Text cleaning
        |
        v
Chunking with metadata
        |
        v
Embedding generation
        |
        v
Vector database storage
        |
        v
Retriever
        |
        v
LLM answer generation
        |
        v
Answer with source references
```

RAG requirements:

- Retrieve relevant filing chunks before answering.
- Include source references in every factual answer.
- Refuse to answer if no relevant source is found.
- Distinguish between facts, assumptions, and model interpretation.
- Prefer primary filings over news summaries for fundamental claims.
- Store metadata for ticker, filing type, filing date, section, and chunk index.

Example RAG questions:

```text
What are the main risk factors mentioned in the latest 10-K?
How does management explain the latest revenue growth?
Did the company mention AI-related capital expenditure?
What changed in the risk factors compared with the previous annual report?
What are the main drivers of gross margin pressure?
```

---

## 11. API Design

### Company Profile

```http
GET /api/stocks/{ticker}
```

### Historical Prices

```http
GET /api/stocks/{ticker}/prices?start=2020-01-01&end=2026-01-01
```

### Financial Ratios

```http
GET /api/stocks/{ticker}/financial-ratios
```

### DCF Valuation

```http
POST /api/stocks/{ticker}/valuation/dcf
```

Request body example:

```json
{
  "revenue_growth": [0.08, 0.06, 0.05, 0.04, 0.03],
  "operating_margin": 0.28,
  "tax_rate": 0.18,
  "wacc": 0.09,
  "terminal_growth": 0.03
}
```

### Filing Q&A

```http
POST /api/stocks/{ticker}/rag/question
```

Request body example:

```json
{
  "question": "What are the main risk factors mentioned in the latest 10-K?"
}
```

Response should include answer and sources.

### Report Generation

```http
POST /api/stocks/{ticker}/reports
```

---

## 12. Frontend Pages

### Streamlit MVP Pages

- Home
- Company Overview
- Financial Analysis
- Valuation Dashboard
- Filing Q&A
- Peer Comparison
- News & Events
- Research Report
- Watchlist

### Advanced Web App Pages

- `/dashboard`
- `/stocks/[ticker]`
- `/stocks/[ticker]/financials`
- `/stocks/[ticker]/valuation`
- `/stocks/[ticker]/filings`
- `/stocks/[ticker]/news`
- `/stocks/[ticker]/peers`
- `/reports/[report_id]`
- `/watchlist`
- `/alerts`

---

## 13. Development Roadmap

### Phase 0: Project Setup

Estimated duration: 1 week

Tasks:

- Create repository.
- Create project structure.
- Add environment configuration.
- Add logging.
- Add pytest setup.
- Add README.
- Add empty FastAPI and Streamlit entry points.

Acceptance criteria:

```text
pytest runs successfully.
streamlit run src/app/streamlit_app.py starts a page.
uvicorn src.api.main:app --reload starts the backend.
```

### Phase 1: Data Ingestion and Persistence

Estimated duration: 2–4 weeks

Tasks:

- Implement market data loader.
- Implement financial data loader.
- Implement database models.
- Store company profiles.
- Store stock prices.
- Store financial statements.
- Add data quality checks.
- Add batch update scripts.

Acceptance criteria:

```text
The system supports at least 10 tickers.
Each ticker has at least 5 years of price data.
Each ticker has at least 5 years of financial data.
Data can be saved and retrieved from the database.
```

### Phase 2: Financial Analysis and Technical Indicators

Estimated duration: 2–3 weeks

Tasks:

- Implement growth metrics.
- Implement margin metrics.
- Implement cash flow metrics.
- Implement ROE, ROA, ROIC.
- Implement leverage and liquidity metrics.
- Implement moving averages.
- Implement RSI.
- Implement volatility and max drawdown.
- Build basic visualizations.

Acceptance criteria:

```text
The app displays financial trends.
The app displays price charts and moving averages.
Financial calculations are covered by unit tests.
```

### Phase 3: Valuation Models

Estimated duration: 3–5 weeks

Tasks:

- Implement PE valuation.
- Implement PS valuation.
- Implement EV/EBITDA valuation.
- Implement DCF.
- Implement Bear/Base/Bull assumptions.
- Implement sensitivity analysis.
- Implement market-implied expectations.
- Build valuation dashboard.

Acceptance criteria:

```text
Each stock supports at least three valuation methods.
DCF produces intrinsic value per share.
Scenario outputs are clearly explained.
Users can adjust assumptions.
```

### Phase 4: SEC Filing Parsing and RAG

Estimated duration: 4–6 weeks

Tasks:

- Implement CIK mapping.
- Fetch filing metadata.
- Download filings.
- Parse filing HTML or text.
- Extract sections.
- Chunk text with metadata.
- Generate embeddings.
- Store vectors.
- Implement retriever.
- Implement filing Q&A.
- Display source references.

Acceptance criteria:

```text
At least 5 companies are supported.
At least one 10-K is parsed per company.
Users can ask filing-based questions.
Every answer includes sources.
The system refuses unsupported answers.
```

### Phase 5: Peer and Industry Analysis

Estimated duration: 3–4 weeks

Tasks:

- Create peer mapping.
- Implement automatic peer suggestions.
- Fetch peer metrics.
- Build peer comparison tables.
- Build growth-versus-valuation charts.
- Generate peer comparison commentary.
- Generate industry summaries.

### Phase 6: News Events and Alerts

Estimated duration: 4–6 weeks

Tasks:

- Implement news ingestion.
- Deduplicate news.
- Classify news events.
- Summarize news.
- Add sentiment scoring.
- Add importance scoring.
- Implement watchlist.
- Implement alert rules.
- Implement alert engine.

### Phase 7: Research Report Generation

Estimated duration: 3–5 weeks

Tasks:

- Design report template.
- Generate company overview.
- Generate financial analysis section.
- Generate valuation section.
- Generate peer comparison section.
- Generate filing summary section.
- Generate news section.
- Generate risk factors.
- Generate monitoring checklist.
- Export Markdown.
- Export PDF.

### Phase 8: Integration and Deployment

Estimated duration: 2–4 weeks

Tasks:

- Clean code structure.
- Add API documentation.
- Improve README.
- Add unit and integration tests.
- Add Docker Compose.
- Prepare demo data.
- Prepare screenshots.
- Prepare sample reports.

---

## 14. Feature Backlog

### P0: Must Have

- Project structure
- Market data loader
- Financial data loader
- Financial ratio calculation
- Technical indicator calculation
- PE valuation
- PS valuation
- Bear/Base/Bull scenario valuation
- Streamlit dashboard
- Markdown report generation

### P1: Core Advanced Features

- FastAPI backend
- PostgreSQL database
- SEC filing download
- 10-K / 10-Q parsing
- Filing RAG Q&A
- DCF valuation
- Peer comparison
- News summary
- PDF export
- Watchlist

### P2: Differentiating Features

- Market-implied expectation analysis
- Filing change detection
- News sentiment analysis
- Event importance scoring
- DCF sensitivity analysis
- Automatic risk alerts
- Industry report generation
- Research Agent workflow

### P3: Long-Term Enhancements

- Next.js frontend
- User authentication
- Multi-market support
- Macro data integration
- Portfolio analysis
- Email or Telegram notifications
- Local LLM support
- Multilingual report generation

---

## 15. Testing Strategy

### Unit Tests

Test:

- Data validation
- Financial ratio formulas
- Valuation formulas
- DCF calculation steps
- Technical indicator formulas
- Filing chunking behavior
- RAG source formatting

### Integration Tests

Test:

- Data ingestion to database
- API response correctness
- End-to-end report generation
- RAG retrieval and answer generation
- Watchlist alert triggering

### Manual Acceptance Tickers

```text
AAPL
MSFT
NVDA
TSLA
AMZN
GOOGL
META
AMD
INTC
JPM
```

---

## 16. Development Principles

1. **Source-grounded analysis first**  
   Do not allow generated conclusions without clear data or source support.

2. **Separate facts from assumptions**  
   Financial data, filing text, model assumptions, and AI interpretations should be clearly separated.

3. **Scenario-based valuation**  
   Avoid single-point price predictions. Use Bear/Base/Bull ranges.

4. **Reusable modules**  
   Every major capability should be implemented as a reusable Python module before being exposed in the UI.

5. **Test financial formulas**  
   All ratio and valuation formulas should have unit tests.

6. **Fail safely**  
   If data is missing, the system should display a clear warning instead of generating misleading conclusions.

7. **No financial advice language**  
   Avoid direct buy/sell/hold commands.

---

## 17. Example Report Structure

```markdown
# {Company Name} Equity Research Report

## 1. Executive Summary
## 2. Business Overview
## 3. Recent Stock Performance
## 4. Financial Performance
## 5. Margin and Cash Flow Analysis
## 6. Balance Sheet Risk
## 7. Industry and Peer Comparison
## 8. SEC Filing Summary
## 9. News and Event Analysis
## 10. Valuation Analysis
### 10.1 PE Valuation
### 10.2 PS Valuation
### 10.3 EV/EBITDA Valuation
### 10.4 DCF Valuation
### 10.5 Bear/Base/Bull Scenarios
### 10.6 Market-Implied Expectations
## 11. Key Risks
## 12. Key Catalysts
## 13. Monitoring Checklist
## 14. Data Sources and Methodology
## 15. Disclaimer
```

---

## 18. Example Local Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
streamlit run src/app/streamlit_app.py
uvicorn src.api.main:app --reload
docker compose up --build
```

---

## 19. Success Criteria

The project can be considered successful when:

```text
1. A user can input a ticker and view company profile, price chart, financial metrics, valuation, and report.
2. The system can parse at least one SEC filing per supported company.
3. The RAG system can answer filing-based questions with source references.
4. The valuation system supports relative valuation and DCF.
5. The system can generate a complete Markdown or PDF research report.
6. The system has tests for core calculations.
7. The project can be run locally from documented instructions.
8. The README clearly explains the project, architecture, usage, and disclaimer.
```

---

## 20. Portfolio / Resume Description

```text
Stock Research Copilot Advanced | Python, FastAPI, Streamlit, PostgreSQL, SEC Filings, RAG, Valuation Modeling

- Built an AI-assisted equity research system that collects market data, financial statements, SEC filings, news, and peer data for public companies.
- Implemented financial ratio analysis, technical indicators, peer comparison, and multi-model valuation including PE, PS, EV/EBITDA, and DCF.
- Developed a retrieval-augmented generation pipeline over SEC filings to support source-grounded financial Q&A.
- Designed an automated research report generator combining structured financial data, valuation outputs, filing summaries, and risk analysis.
- Added watchlist and alerting logic for price moves, new filings, valuation deviations, and important company events.
```
