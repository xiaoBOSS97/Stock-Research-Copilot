from __future__ import annotations

from src.data_loader.sec_loader import (
    SecFiling,
    build_filing_url,
    download_filing,
    get_recent_filings,
    lookup_cik,
    normalize_cik,
    parse_recent_filings,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object] | None = None, text: str = "") -> None:
        self.payload = payload or {}
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class FakeSecSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, headers: dict[str, str], timeout: int) -> FakeResponse:
        self.calls.append(url)
        if url.endswith("company_tickers.json"):
            return FakeResponse(
                {
                    "0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."},
                    "1": {"ticker": "MSFT", "cik_str": 789019, "title": "Microsoft Corp"},
                }
            )
        if "CIK0000320193.json" in url:
            return FakeResponse(sample_submissions_payload())
        return FakeResponse(text="<html>filing text</html>")


def sample_submissions_payload() -> dict[str, object]:
    return {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-25-000079", "0000320193-25-000008"],
                "filingDate": ["2025-10-31", "2025-01-31"],
                "reportDate": ["2025-09-27", "2024-12-28"],
                "form": ["10-K", "10-Q"],
                "primaryDocument": ["aapl-20250927.htm", "aapl-20241228.htm"],
            }
        },
    }


def test_normalize_cik_zero_pads_to_ten_digits() -> None:
    assert normalize_cik(320193) == "0000320193"


def test_lookup_cik_uses_sec_ticker_mapping() -> None:
    session = FakeSecSession()

    assert lookup_cik(" aapl ", session=session, user_agent="test@example.com") == "0000320193"


def test_build_filing_url_uses_compact_accession_number() -> None:
    assert build_filing_url("0000320193", "0000320193-25-000079", "aapl-20250927.htm") == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
    )


def test_parse_recent_filings_filters_forms_and_builds_urls() -> None:
    filings = parse_recent_filings(sample_submissions_payload(), ticker="AAPL", forms=("10-K",), limit=5)

    assert len(filings) == 1
    assert filings[0].form == "10-K"
    assert filings[0].company_name == "Apple Inc."
    assert filings[0].file_url.endswith("/000032019325000079/aapl-20250927.htm")


def test_get_recent_filings_combines_mapping_and_submissions() -> None:
    session = FakeSecSession()

    filings = get_recent_filings("AAPL", forms=("10-K", "10-Q"), limit=2, session=session)

    assert [filing.form for filing in filings] == ["10-K", "10-Q"]
    assert any("company_tickers.json" in call for call in session.calls)
    assert any("CIK0000320193.json" in call for call in session.calls)


def test_download_filing_saves_primary_document(tmp_path) -> None:
    filing = SecFiling(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form="10-K",
        filing_date="2025-10-31",
        report_date="2025-09-27",
        accession_number="0000320193-25-000079",
        primary_document="aapl-20250927.htm",
        file_url="https://example.test/aapl-20250927.htm",
    )
    session = FakeSecSession()

    path = download_filing(filing, output_dir=tmp_path, session=session)

    assert path.exists()
    assert path.read_text(encoding="utf-8") == "<html>filing text</html>"
    assert path.parent == tmp_path / "AAPL"
