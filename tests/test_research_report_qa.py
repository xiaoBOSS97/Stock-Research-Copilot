from __future__ import annotations

from src.rag.research_report_qa import (
    analyze_research_report,
    chunk_research_report,
    search_research_report,
    summarize_research_report,
)


REPORT_TEXT = """
The analyst maintains an Outperform rating and raises the price target to 250.
The valuation argument relies on revenue growth, margin expansion, and stronger cash flow.
Key risks include competition, regulatory scrutiny, and execution risk in new products.
"""


def test_chunk_research_report_splits_text() -> None:
    chunks = chunk_research_report("A" * 2000, chunk_size=1000, overlap=100)

    assert len(chunks) == 3
    assert chunks[0].chunk_id == 1


def test_search_research_report_returns_relevant_passage() -> None:
    results = search_research_report(REPORT_TEXT, "What is the price target and rating?")

    assert not results.empty
    assert "price target" in results.iloc[0]["Passage"].lower()
    assert "outperform" in results.iloc[0]["Passage"].lower()


def test_analyze_research_report_finds_topics() -> None:
    analysis = analyze_research_report(REPORT_TEXT)

    assert analysis["word_count"] > 0
    assert not analysis["topic_table"].empty
    assert "Rating / Recommendation" in analysis["topic_table"]["Topic"].tolist()
    assert analysis["summary"]


def test_summarize_research_report_returns_bullets() -> None:
    summary = summarize_research_report(REPORT_TEXT)

    assert "Rating / Recommendation" in summary
    assert "Price Target / Valuation" in summary
