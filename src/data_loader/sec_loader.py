"""SEC EDGAR filing metadata and download helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
from typing import Any, Iterable

from dotenv import load_dotenv
import requests


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_SEC_USER_AGENT = "StockResearchCopilot/0.1 educational project contact@example.com"
DEFAULT_FILING_FORMS = ("10-K", "10-Q", "8-K")


class SecFilingError(RuntimeError):
    """Raised when SEC filing data cannot be retrieved or parsed."""


@dataclass(frozen=True)
class SecFiling:
    """Normalized SEC filing metadata used by the app and RAG pipeline."""

    ticker: str
    cik: str
    company_name: str | None
    form: str
    filing_date: str
    report_date: str | None
    accession_number: str
    primary_document: str
    file_url: str
    local_path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return a stable dictionary representation for tables and reports."""

        return asdict(self)


def get_sec_user_agent() -> str:
    """Read SEC_USER_AGENT from the environment, falling back to a local default."""

    load_dotenv()
    user_agent = os.getenv("SEC_USER_AGENT")
    return user_agent.strip() if user_agent else DEFAULT_SEC_USER_AGENT


def sec_headers(user_agent: str | None = None) -> dict[str, str]:
    """Return headers accepted by SEC EDGAR JSON and archive endpoints."""

    return {
        "User-Agent": user_agent or get_sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
    }


def normalize_ticker(ticker: str) -> str:
    """Normalize a stock ticker for SEC lookup."""

    if not isinstance(ticker, str):
        raise TypeError("Ticker must be a string.")
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker must not be empty.")
    return symbol


def normalize_cik(cik: int | str) -> str:
    """Return a zero-padded 10 digit CIK string."""

    cik_text = str(cik).strip()
    if not re.fullmatch(r"\d{1,10}", cik_text):
        raise ValueError("CIK must contain 1 to 10 digits.")
    return cik_text.zfill(10)


def cik_without_leading_zeroes(cik: int | str) -> str:
    """Return the archive URL CIK form without leading zeroes."""

    return str(int(normalize_cik(cik)))


def get_ticker_cik_map(
    *,
    session: Any = requests,
    timeout: int = 30,
    user_agent: str | None = None,
) -> dict[str, dict[str, str]]:
    """Download the SEC ticker-to-CIK mapping."""

    try:
        response = session.get(
            SEC_COMPANY_TICKERS_URL,
            headers=sec_headers(user_agent),
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise SecFilingError(f"Could not download SEC ticker mapping: {exc}") from exc
    except ValueError as exc:
        raise SecFilingError("SEC ticker mapping returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise SecFilingError("SEC ticker mapping has an unexpected shape.")

    ticker_map: dict[str, dict[str, str]] = {}
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        ticker = item.get("ticker")
        cik = item.get("cik_str")
        title = item.get("title")
        if ticker is None or cik is None:
            continue
        ticker_map[str(ticker).upper()] = {
            "ticker": str(ticker).upper(),
            "cik": normalize_cik(cik),
            "title": str(title) if title is not None else "",
        }
    return ticker_map


def lookup_cik(
    ticker: str,
    *,
    session: Any = requests,
    timeout: int = 30,
    user_agent: str | None = None,
) -> str:
    """Look up a ticker's SEC CIK."""

    symbol = normalize_ticker(ticker)
    mapping = get_ticker_cik_map(session=session, timeout=timeout, user_agent=user_agent)
    if symbol not in mapping:
        raise SecFilingError(f"No SEC CIK found for ticker {symbol}.")
    return mapping[symbol]["cik"]


def get_company_submissions(
    cik: int | str,
    *,
    session: Any = requests,
    timeout: int = 30,
    user_agent: str | None = None,
) -> dict[str, Any]:
    """Download a company's SEC submissions JSON payload."""

    normalized_cik = normalize_cik(cik)
    try:
        response = session.get(
            SEC_SUBMISSIONS_URL.format(cik=normalized_cik),
            headers=sec_headers(user_agent),
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise SecFilingError(f"Could not download SEC submissions for CIK {normalized_cik}: {exc}") from exc
    except ValueError as exc:
        raise SecFilingError("SEC submissions endpoint returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise SecFilingError("SEC submissions payload has an unexpected shape.")
    return payload


def parse_recent_filings(
    submissions_payload: dict[str, Any],
    *,
    ticker: str,
    cik: int | str | None = None,
    forms: Iterable[str] | None = DEFAULT_FILING_FORMS,
    limit: int | None = None,
) -> list[SecFiling]:
    """Normalize recent SEC filing metadata from a submissions payload."""

    symbol = normalize_ticker(ticker)
    normalized_cik = normalize_cik(cik or submissions_payload.get("cik", ""))
    company_name = submissions_payload.get("name")
    recent = submissions_payload.get("filings", {}).get("recent")
    if not isinstance(recent, dict):
        raise SecFilingError("SEC submissions payload is missing recent filings.")

    accession_numbers = _recent_column(recent, "accessionNumber")
    filing_dates = _recent_column(recent, "filingDate")
    forms_column = _recent_column(recent, "form")
    primary_documents = _recent_column(recent, "primaryDocument")
    report_dates = recent.get("reportDate", [])
    allowed_forms = {form.upper() for form in forms} if forms is not None else None

    filings: list[SecFiling] = []
    row_count = min(len(accession_numbers), len(filing_dates), len(forms_column), len(primary_documents))
    for index in range(row_count):
        form = str(forms_column[index]).upper()
        if allowed_forms is not None and form not in allowed_forms:
            continue
        accession_number = str(accession_numbers[index])
        primary_document = str(primary_documents[index])
        filing = SecFiling(
            ticker=symbol,
            cik=normalized_cik,
            company_name=str(company_name) if company_name is not None else None,
            form=form,
            filing_date=str(filing_dates[index]),
            report_date=_optional_string_at(report_dates, index),
            accession_number=accession_number,
            primary_document=primary_document,
            file_url=build_filing_url(normalized_cik, accession_number, primary_document),
        )
        filings.append(filing)
        if limit is not None and len(filings) >= limit:
            break

    return filings


def get_recent_filings(
    ticker: str,
    *,
    forms: Iterable[str] | None = DEFAULT_FILING_FORMS,
    limit: int | None = 10,
    session: Any = requests,
    timeout: int = 30,
    user_agent: str | None = None,
) -> list[SecFiling]:
    """Look up a ticker and return its recent SEC filings."""

    symbol = normalize_ticker(ticker)
    ticker_map = get_ticker_cik_map(session=session, timeout=timeout, user_agent=user_agent)
    if symbol not in ticker_map:
        raise SecFilingError(f"No SEC CIK found for ticker {symbol}.")
    cik = ticker_map[symbol]["cik"]
    payload = get_company_submissions(cik, session=session, timeout=timeout, user_agent=user_agent)
    return parse_recent_filings(payload, ticker=symbol, cik=cik, forms=forms, limit=limit)


def get_latest_filing(
    ticker: str,
    form: str = "10-K",
    *,
    session: Any = requests,
    timeout: int = 30,
    user_agent: str | None = None,
) -> SecFiling:
    """Return the latest filing for a ticker and form."""

    filings = get_recent_filings(
        ticker,
        forms=(form,),
        limit=1,
        session=session,
        timeout=timeout,
        user_agent=user_agent,
    )
    if not filings:
        raise SecFilingError(f"No recent {form.upper()} filing found for {normalize_ticker(ticker)}.")
    return filings[0]


def build_filing_url(cik: int | str, accession_number: str, primary_document: str) -> str:
    """Build the SEC archive URL for a filing primary document."""

    accession_compact = accession_number.replace("-", "")
    return (
        f"{SEC_ARCHIVES_BASE_URL}/{cik_without_leading_zeroes(cik)}/"
        f"{accession_compact}/{primary_document}"
    )


def download_filing(
    filing: SecFiling,
    *,
    output_dir: str | Path = "data/filings",
    session: Any = requests,
    timeout: int = 30,
    user_agent: str | None = None,
) -> Path:
    """Download a filing's primary document and save it locally."""

    try:
        response = session.get(
            filing.file_url,
            headers=sec_headers(user_agent),
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.text
    except requests.RequestException as exc:
        raise SecFilingError(f"Could not download SEC filing {filing.accession_number}: {exc}") from exc

    target_path = local_filing_path(filing, output_dir=output_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return target_path


def download_latest_filing(
    ticker: str,
    form: str = "10-K",
    *,
    output_dir: str | Path = "data/filings",
    session: Any = requests,
    timeout: int = 30,
    user_agent: str | None = None,
) -> Path:
    """Look up and download the latest filing for a ticker and form."""

    filing = get_latest_filing(ticker, form=form, session=session, timeout=timeout, user_agent=user_agent)
    return download_filing(
        filing,
        output_dir=output_dir,
        session=session,
        timeout=timeout,
        user_agent=user_agent,
    )


def local_filing_path(filing: SecFiling, *, output_dir: str | Path = "data/filings") -> Path:
    """Return the deterministic local path for a filing primary document."""

    filename = "_".join(
        [
            filing.form.replace("/", "-"),
            filing.filing_date,
            filing.accession_number,
            _safe_filename(filing.primary_document),
        ]
    )
    return Path(output_dir) / filing.ticker / filename


def _recent_column(recent: dict[str, Any], key: str) -> list[Any]:
    values = recent.get(key)
    if not isinstance(values, list):
        raise SecFilingError(f"SEC submissions recent filings missing {key}.")
    return values


def _optional_string_at(values: Any, index: int) -> str | None:
    if not isinstance(values, list) or index >= len(values):
        return None
    value = values[index]
    if value is None or value == "":
        return None
    return str(value)


def _safe_filename(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or "filing.txt"
