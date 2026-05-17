"""Local financial organization report upload and extraction helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

from src.rag.research_report_qa import clean_research_report_text


DEFAULT_RESEARCH_REPORT_DIR = Path("data/research_reports")
SUPPORTED_RESEARCH_REPORT_SUFFIXES = {".txt", ".md", ".pdf"}


class ResearchReportError(RuntimeError):
    """Raised when an external research report cannot be loaded or parsed."""


def extract_research_report_text(file_name: str, content: bytes) -> str:
    """Extract readable text from an uploaded text, Markdown, or PDF report."""

    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_RESEARCH_REPORT_SUFFIXES:
        raise ValueError("Research reports must be .txt, .md, or .pdf.")
    if suffix == ".pdf":
        return extract_pdf_text(content)
    return clean_research_report_text(content.decode("utf-8", errors="replace"))


def extract_pdf_text(content: bytes) -> str:
    """Extract text from a PDF using pypdf when available."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ResearchReportError("PDF extraction requires pypdf. Run `uv sync` first.") from exc

    try:
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ResearchReportError(f"Could not extract text from PDF report: {exc}") from exc

    text = clean_research_report_text("\n\n".join(pages))
    if not text:
        raise ResearchReportError("No extractable text found in the PDF report.")
    return text


def save_research_report_text(
    ticker: str,
    file_name: str,
    text: str,
    output_dir: str | Path = DEFAULT_RESEARCH_REPORT_DIR,
) -> Path:
    """Save extracted report text under ``data/research_reports/{TICKER}/``."""

    symbol = normalize_ticker(ticker)
    safe_stem = _safe_stem(file_name) or "research_report"
    target_dir = Path(output_dir) / symbol
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_stem}.txt"
    target.write_text(clean_research_report_text(text) + "\n", encoding="utf-8")
    return target


def save_uploaded_research_report(
    ticker: str,
    file_name: str,
    content: bytes,
    output_dir: str | Path = DEFAULT_RESEARCH_REPORT_DIR,
) -> Path:
    """Extract and save an uploaded financial organization report."""

    text = extract_research_report_text(file_name, content)
    return save_research_report_text(ticker, file_name, text, output_dir=output_dir)


def load_research_report_text(path: str | Path) -> str:
    """Load an extracted local research report text file."""

    source = Path(path)
    if source.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Saved research report text must be .txt or .md.")
    return clean_research_report_text(source.read_text(encoding="utf-8"))


def list_research_report_files(
    ticker: str,
    report_dir: str | Path = DEFAULT_RESEARCH_REPORT_DIR,
) -> list[Path]:
    """Return extracted financial organization reports for a ticker, newest first."""

    symbol = normalize_ticker(ticker)
    source_dir = Path(report_dir) / symbol
    if not source_dir.exists():
        return []
    files = [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    ]
    return sorted(files, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def normalize_ticker(ticker: str) -> str:
    """Normalize a ticker symbol for local storage."""

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker must not be empty.")
    return symbol


def _safe_stem(file_name: str) -> str:
    stem = Path(file_name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
