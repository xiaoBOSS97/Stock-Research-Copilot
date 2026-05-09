from __future__ import annotations

import json

from src.preprocessing.filing_parser import (
    chunk_filing_sections,
    extract_filing_sections,
    html_to_text,
    load_filing_text,
    parse_filing_file,
    save_parsed_filing,
)


SAMPLE_FILING_HTML = """
<html>
  <head><style>.hidden { display: none; }</style></head>
  <body>
    <p>Item 1. Business</p>
    <p>We design consumer technology products and services.</p>
    <p>Item 1A. Risk Factors</p>
    <p>Our business faces supply chain risk and competitive pressure.</p>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>Revenue increased because demand was strong across customers.</p>
    <p>Item 8. Financial Statements</p>
    <p>Consolidated statements are included in this filing.</p>
  </body>
</html>
"""


def test_html_to_text_strips_tags_and_preserves_content() -> None:
    text = html_to_text(SAMPLE_FILING_HTML)

    assert "<p>" not in text
    assert "Item 1. Business" in text
    assert "supply chain risk" in text


def test_extract_filing_sections_finds_major_sec_items() -> None:
    sections = extract_filing_sections(html_to_text(SAMPLE_FILING_HTML))
    section_names = [section.section for section in sections]

    assert "Business" in section_names
    assert "Risk Factors" in section_names
    assert "Management Discussion and Analysis" in section_names
    risk_section = next(section for section in sections if section.section == "Risk Factors")
    assert risk_section.item == "1A"
    assert "competitive pressure" in risk_section.text


def test_parse_filing_file_loads_html_from_disk(tmp_path) -> None:
    filing_path = tmp_path / "sample.htm"
    filing_path.write_text(SAMPLE_FILING_HTML, encoding="utf-8")

    sections = parse_filing_file(filing_path)

    assert sections[0].section == "Business"
    assert "consumer technology" in sections[0].text


def test_load_filing_text_accepts_plain_text_filings(tmp_path) -> None:
    filing_path = tmp_path / "sample.txt"
    filing_path.write_text("Item 1. Business\nPlain text business section.", encoding="utf-8")

    assert "Plain text business" in load_filing_text(filing_path)


def test_chunk_filing_sections_creates_source_refs() -> None:
    sections = extract_filing_sections(html_to_text(SAMPLE_FILING_HTML))

    chunks = chunk_filing_sections(
        sections,
        ticker="AAPL",
        filing_type="10-K",
        filing_date="2025-10-31",
        chunk_size=80,
        overlap=10,
    )

    assert len(chunks) > 1
    assert chunks[0].chunk_id == 1
    assert "AAPL | 10-K | 2025-10-31" in chunks[0].source_ref


def test_extract_filing_sections_falls_back_to_full_text_when_no_items() -> None:
    sections = extract_filing_sections("No recognizable SEC item headings here.")

    assert len(sections) == 1
    assert sections[0].section == "Filing Text"
    assert sections[0].item is None


def test_save_parsed_filing_writes_json(tmp_path) -> None:
    sections = extract_filing_sections(html_to_text(SAMPLE_FILING_HTML))

    output = save_parsed_filing(
        sections,
        ticker="aapl",
        accession_number="0000320193-25-000079",
        output_dir=tmp_path,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.parent.name == "AAPL"
    assert payload["ticker"] == "AAPL"
    assert payload["sections"][0]["section"] == "Business"
