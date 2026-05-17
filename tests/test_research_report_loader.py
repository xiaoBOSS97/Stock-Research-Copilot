from __future__ import annotations

import pytest

from src.data_loader.research_report_loader import (
    extract_research_report_text,
    list_research_report_files,
    load_research_report_text,
    save_uploaded_research_report,
)


def test_extract_research_report_text_supports_text_upload() -> None:
    text = extract_research_report_text(
        "bank_report.txt",
        b" Rating: Buy. Price target is 200. Revenue growth remains strong. ",
    )

    assert text == "Rating: Buy. Price target is 200. Revenue growth remains strong."


def test_extract_research_report_text_rejects_unknown_suffix() -> None:
    with pytest.raises(ValueError, match=".txt, .md, or .pdf"):
        extract_research_report_text("report.docx", b"content")


def test_save_uploaded_research_report_sanitizes_and_lists(tmp_path) -> None:
    saved = save_uploaded_research_report(
        "aapl",
        "Big Bank Report!.md",
        b"Analyst rating is outperform. The report discusses margin risk.",
        output_dir=tmp_path,
    )

    files = list_research_report_files("AAPL", tmp_path)

    assert saved == tmp_path / "AAPL" / "Big_Bank_Report.txt"
    assert files == [saved]
    assert "outperform" in load_research_report_text(saved)
