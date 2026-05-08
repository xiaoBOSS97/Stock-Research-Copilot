from __future__ import annotations

from src.rag.transcript_analysis import (
    analyze_transcript,
    chunk_transcript,
    list_transcript_files,
    load_transcript_text,
    save_transcript_text,
    search_transcript,
)


SAMPLE_TRANSCRIPT = """
Operator: Welcome to the quarterly earnings call.

CEO: Demand remained strong across enterprise customers, and revenue growth improved.
Gross margin expanded because pricing was favorable and supply costs improved.
CFO: For next quarter, our guidance assumes continued growth but some macro headwinds.
Analyst: What risks are you watching?
CEO: Competition and inventory remain important risks.
"""


def test_save_load_and_list_transcript(tmp_path) -> None:
    saved = save_transcript_text("nvda", "Q1 Call.txt", SAMPLE_TRANSCRIPT, output_dir=tmp_path)

    assert saved.name == "Q1_Call.txt"
    assert saved.parent.name == "NVDA"
    assert list_transcript_files("NVDA", transcript_dir=tmp_path) == [saved]
    assert "Demand remained strong" in load_transcript_text(saved)


def test_chunk_transcript_creates_overlapping_chunks() -> None:
    chunks = chunk_transcript("abc " * 400, chunk_size=200, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == 1
    assert chunks[0].text


def test_analyze_transcript_returns_topic_signals() -> None:
    analysis = analyze_transcript(SAMPLE_TRANSCRIPT)
    topic_table = analysis["topic_table"]

    assert analysis["word_count"] > 0
    assert analysis["management_tone"] in {"Constructive", "Balanced", "Cautious"}
    assert "Revenue / Demand" in topic_table["Topic"].tolist()
    assert topic_table.loc[topic_table["Topic"] == "Revenue / Demand", "Mentions"].iloc[0] > 0
    assert analysis["key_questions"]


def test_search_transcript_returns_relevant_passages() -> None:
    results = search_transcript(SAMPLE_TRANSCRIPT, "margin pricing costs", top_k=2)

    assert not results.empty
    assert "Gross margin" in results.iloc[0]["Passage"]
