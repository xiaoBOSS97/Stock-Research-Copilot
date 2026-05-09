from __future__ import annotations

from src.preprocessing.filing_parser import extract_filing_sections, html_to_text, save_parsed_filing
from src.rag.filing_qa import (
    answer_filing_question,
    list_parsed_filing_files,
    load_parsed_filing_sections,
    query_terms,
    search_filing_sections,
)


SAMPLE_FILING_HTML = """
<html>
  <body>
    <p>Item 1. Business</p>
    <p>The company sells devices, software, and cloud services to customers globally.</p>
    <p>Item 1A. Risk Factors</p>
    <p>Supply chain disruption could increase costs and delay product availability.</p>
    <p>Competition may pressure pricing and reduce market share.</p>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>Revenue increased because enterprise demand remained strong.</p>
    <p>Gross margin improved due to favorable product mix.</p>
    <p>Item 8. Financial Statements</p>
    <p>Financial statements are included in this annual report.</p>
  </body>
</html>
"""


def sample_sections():
    return extract_filing_sections(html_to_text(SAMPLE_FILING_HTML))


def test_query_terms_removes_common_words() -> None:
    assert query_terms("What are the main risk factors?") == ["risk", "factor"]


def test_search_filing_sections_returns_relevant_sources() -> None:
    sources = search_filing_sections(sample_sections(), "What risks affect supply chain?", top_k=2)

    assert sources
    assert sources[0].section == "Risk Factors"
    assert "Supply chain disruption" in sources[0].passage


def test_answer_filing_question_returns_cited_answer() -> None:
    result = answer_filing_question(sample_sections(), "Why did revenue increase?", ticker="AAPL", filing_type="10-K")

    assert not result.refused
    assert "Based only on the retrieved filing passages" in result.answer
    assert "[1]" in result.answer
    assert "enterprise demand" in result.answer
    assert not result.sources_frame().empty


def test_answer_filing_question_refuses_without_supporting_source() -> None:
    result = answer_filing_question(sample_sections(), "What did the company say about blockchain?")

    assert result.refused
    assert not result.sources
    assert "could not find a supporting passage" in result.answer


def test_load_and_list_parsed_filing_sections(tmp_path) -> None:
    saved = save_parsed_filing(
        sample_sections(),
        ticker="aapl",
        accession_number="0000320193-25-000079",
        output_dir=tmp_path,
    )

    files = list_parsed_filing_files("AAPL", parsed_dir=tmp_path)
    loaded_sections = load_parsed_filing_sections(saved)

    assert files == [saved]
    assert loaded_sections[0].section == "Business"
    assert "devices" in loaded_sections[0].text
