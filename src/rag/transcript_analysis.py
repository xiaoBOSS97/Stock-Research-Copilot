"""Local earnings call transcript analysis helpers.

This module intentionally starts with deterministic retrieval and keyword
summaries. It gives the app useful transcript analysis without requiring API
keys, embeddings, or network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd


DEFAULT_TRANSCRIPT_DIR = Path("data/transcripts")
SUPPORTED_TRANSCRIPT_SUFFIXES = {".txt", ".md"}
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Revenue / Demand": (
        "revenue",
        "sales",
        "demand",
        "bookings",
        "orders",
        "customer",
        "growth",
    ),
    "Margins / Costs": (
        "margin",
        "gross margin",
        "operating margin",
        "cost",
        "expense",
        "pricing",
    ),
    "Guidance / Outlook": (
        "guidance",
        "outlook",
        "expect",
        "forecast",
        "next quarter",
        "full year",
    ),
    "Risks / Headwinds": (
        "risk",
        "headwind",
        "uncertain",
        "competition",
        "supply",
        "macro",
        "inventory",
    ),
    "Cash Flow / Capital": (
        "cash flow",
        "free cash flow",
        "capex",
        "capital expenditure",
        "buyback",
        "dividend",
    ),
}


@dataclass(frozen=True)
class TranscriptChunk:
    """A searchable transcript text chunk."""

    chunk_id: int
    text: str


def clean_transcript_text(text: str) -> str:
    """Normalize transcript text for storage, chunking, and search."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_transcript_text(path: str | Path) -> str:
    """Load a plain-text or Markdown transcript from disk."""

    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_TRANSCRIPT_SUFFIXES:
        raise ValueError("Transcript files must be .txt or .md.")
    return clean_transcript_text(source.read_text(encoding="utf-8"))


def save_transcript_text(
    ticker: str,
    file_name: str,
    text: str,
    output_dir: str | Path = DEFAULT_TRANSCRIPT_DIR,
) -> Path:
    """Save uploaded transcript text under ``data/transcripts/{TICKER}/``."""

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker must not be empty.")
    suffix = Path(file_name).suffix.lower() or ".txt"
    if suffix not in SUPPORTED_TRANSCRIPT_SUFFIXES:
        suffix = ".txt"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(file_name).stem).strip("._") or "transcript"
    target_dir = Path(output_dir) / symbol
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_stem}{suffix}"
    target.write_text(clean_transcript_text(text) + "\n", encoding="utf-8")
    return target


def list_transcript_files(
    ticker: str,
    transcript_dir: str | Path = DEFAULT_TRANSCRIPT_DIR,
) -> list[Path]:
    """Return locally saved transcript files for a ticker, newest first."""

    symbol = ticker.strip().upper()
    source_dir = Path(transcript_dir) / symbol
    if not source_dir.exists():
        return []
    files = [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_TRANSCRIPT_SUFFIXES
    ]
    return sorted(files, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def chunk_transcript(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[TranscriptChunk]:
    """Split transcript text into overlapping searchable chunks."""

    cleaned = clean_transcript_text(text)
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks = []
    start = 0
    chunk_id = 1
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(TranscriptChunk(chunk_id=chunk_id, text=cleaned[start:end].strip()))
        if end == len(cleaned):
            break
        start = end - overlap
        chunk_id += 1
    return chunks


def search_transcript(
    text: str,
    query: str,
    top_k: int = 5,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> pd.DataFrame:
    """Return top transcript chunks for a plain-language query."""

    terms = _query_terms(query)
    chunks = chunk_transcript(text, chunk_size=chunk_size, overlap=overlap)
    rows = []
    for chunk in chunks:
        score = _score_text(chunk.text, terms)
        if score <= 0:
            continue
        rows.append(
            {
                "Chunk": chunk.chunk_id,
                "Score": score,
                "Passage": _compact_text(chunk.text),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Chunk", "Score", "Passage"])
    return pd.DataFrame(rows).sort_values(["Score", "Chunk"], ascending=[False, True]).head(top_k)


def analyze_transcript(text: str) -> dict[str, Any]:
    """Build a compact transcript analysis pack for the dashboard."""

    cleaned = clean_transcript_text(text)
    sentences = _split_sentences(cleaned)
    topic_rows = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        matches = _matching_sentences(sentences, keywords)
        topic_rows.append(
            {
                "Topic": topic,
                "Mentions": len(matches),
                "Representative Comment": matches[0] if matches else "N/A",
            }
        )

    return {
        "word_count": len(re.findall(r"\b\w+\b", cleaned)),
        "chunk_count": len(chunk_transcript(cleaned)),
        "topic_table": pd.DataFrame(topic_rows),
        "management_tone": _management_tone(cleaned),
        "key_questions": _suggest_questions(topic_rows),
    }


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query.lower())
    stopwords = {
        "the",
        "and",
        "for",
        "are",
        "what",
        "why",
        "how",
        "did",
        "does",
        "was",
        "were",
        "company",
    }
    return [term for term in terms if term not in stopwords]


def _score_text(text: str, terms: list[str]) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in terms)


def _compact_text(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    return [part.strip() for part in parts if part.strip()]


def _matching_sentences(sentences: list[str], keywords: tuple[str, ...]) -> list[str]:
    matches = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            matches.append(_compact_text(sentence, limit=260))
    return matches


def _management_tone(text: str) -> str:
    lower = text.lower()
    positive = sum(lower.count(term) for term in ("strong", "growth", "record", "improve", "demand"))
    negative = sum(lower.count(term) for term in ("decline", "weak", "headwind", "risk", "pressure"))
    if positive > negative * 1.5:
        return "Constructive"
    if negative > positive * 1.5:
        return "Cautious"
    return "Balanced"


def _suggest_questions(topic_rows: list[dict[str, Any]]) -> list[str]:
    active_topics = [row["Topic"] for row in topic_rows if row["Mentions"] > 0]
    questions = []
    if "Revenue / Demand" in active_topics:
        questions.append("What did management say about demand and revenue growth?")
    if "Margins / Costs" in active_topics:
        questions.append("What drove margin or cost changes?")
    if "Guidance / Outlook" in active_topics:
        questions.append("What guidance or outlook did management provide?")
    if "Risks / Headwinds" in active_topics:
        questions.append("What risks or headwinds were discussed?")
    return questions[:4]
