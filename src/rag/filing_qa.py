"""Source-grounded SEC filing question answering helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.preprocessing.filing_parser import (
    DEFAULT_PARSED_FILING_DIR,
    FilingChunk,
    FilingSection,
    chunk_filing_sections,
    clean_filing_text,
)


QUESTION_STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "company",
    "could",
    "did",
    "does",
    "explain",
    "filing",
    "for",
    "from",
    "has",
    "have",
    "how",
    "its",
    "main",
    "mention",
    "mentioned",
    "the",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


@dataclass(frozen=True)
class FilingSource:
    """A cited source passage from a filing."""

    source_id: int
    section: str
    item: str | None
    source_ref: str
    passage: str
    score: int

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a serializable representation of the source."""

        return asdict(self)


@dataclass(frozen=True)
class FilingAnswer:
    """A source-grounded filing Q&A result."""

    question: str
    answer: str
    sources: list[FilingSource]
    refused: bool = False

    def sources_frame(self) -> pd.DataFrame:
        """Return sources in a dashboard-friendly table."""

        if not self.sources:
            return pd.DataFrame(columns=["Source", "Section", "Item", "Score", "Passage"])
        return pd.DataFrame(
            {
                "Source": f"[{source.source_id}] {source.source_ref}",
                "Section": source.section,
                "Item": source.item or "N/A",
                "Score": source.score,
                "Passage": source.passage,
            }
            for source in self.sources
        )


def list_parsed_filing_files(
    ticker: str,
    parsed_dir: str | Path = DEFAULT_PARSED_FILING_DIR,
) -> list[Path]:
    """Return parsed filing JSON files for a ticker, newest first."""

    symbol = ticker.strip().upper()
    source_dir = Path(parsed_dir) / symbol
    if not source_dir.exists():
        return []
    files = [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"]
    return sorted(files, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def load_parsed_filing_sections(path: str | Path) -> list[FilingSection]:
    """Load parsed filing sections saved by ``save_parsed_filing``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sections = payload.get("sections", [])
    if not isinstance(sections, list):
        raise ValueError("Parsed filing JSON is missing a sections list.")

    loaded_sections: list[FilingSection] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        loaded_sections.append(
            FilingSection(
                section=str(section.get("section") or "Filing Text"),
                item=None if section.get("item") is None else str(section.get("item")),
                text=str(section.get("text") or ""),
                start_char=int(section.get("start_char") or 0),
                end_char=int(section.get("end_char") or 0),
            )
        )
    return loaded_sections


def search_filing_sections(
    sections: list[FilingSection],
    question: str,
    *,
    ticker: str | None = None,
    filing_type: str | None = None,
    filing_date: str | None = None,
    top_k: int = 5,
    chunk_size: int = 1800,
    overlap: int = 200,
) -> list[FilingSource]:
    """Retrieve filing passages relevant to a question."""

    terms = query_terms(question)
    if not terms:
        return []

    chunks = chunk_filing_sections(
        sections,
        ticker=ticker,
        filing_type=filing_type,
        filing_date=filing_date,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    scored: list[tuple[int, FilingChunk]] = []
    for chunk in chunks:
        score = score_text(chunk.text, terms)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    sources = []
    for source_id, (score, chunk) in enumerate(scored[:top_k], start=1):
        sources.append(
            FilingSource(
                source_id=source_id,
                section=chunk.section,
                item=chunk.item,
                source_ref=chunk.source_ref,
                passage=best_passage(chunk.text, terms),
                score=score,
            )
        )
    return sources


def answer_filing_question(
    sections: list[FilingSection],
    question: str,
    *,
    ticker: str | None = None,
    filing_type: str | None = None,
    filing_date: str | None = None,
    top_k: int = 5,
) -> FilingAnswer:
    """Answer a filing question using only retrieved source passages."""

    cleaned_question = question.strip()
    if not cleaned_question:
        return FilingAnswer(
            question=question,
            answer="Ask a specific question about the parsed filing.",
            sources=[],
            refused=True,
        )

    sources = search_filing_sections(
        sections,
        cleaned_question,
        ticker=ticker,
        filing_type=filing_type,
        filing_date=filing_date,
        top_k=top_k,
    )
    if not sources:
        return FilingAnswer(
            question=cleaned_question,
            answer=(
                "I could not find a supporting passage in the parsed filing for that question. "
                "Try asking about a specific section such as risks, revenue, margins, cash flow, "
                "competition, or management discussion."
            ),
            sources=[],
            refused=True,
        )

    answer_lines = []
    for source in sources[:3]:
        answer_lines.append(f"- {source.passage} [{source.source_id}]")
    answer = (
        "Based only on the retrieved filing passages:\n"
        + "\n".join(answer_lines)
        + "\n\nThis is a source-grounded summary, not investment advice."
    )
    return FilingAnswer(question=cleaned_question, answer=answer, sources=sources)


def query_terms(question: str) -> list[str]:
    """Extract useful retrieval terms from a filing question."""

    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", question.lower())
    terms = []
    for term in raw_terms:
        singular = term[:-1] if term.endswith("s") and len(term) > 4 else term
        if singular not in QUESTION_STOPWORDS and singular not in terms:
            terms.append(singular)
    return terms


def score_text(text: str, terms: list[str]) -> int:
    """Score text by simple term frequency."""

    lower = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(term)}\w*\b", lower)) for term in terms)


def best_passage(text: str, terms: list[str], *, limit: int = 600) -> str:
    """Return the most relevant compact passage from a chunk."""

    sentences = split_sentences(text)
    if not sentences:
        return compact_text(text, limit=limit)
    ranked = sorted(sentences, key=lambda sentence: score_text(sentence, terms), reverse=True)
    selected = []
    total_length = 0
    for sentence in ranked:
        sentence_score = score_text(sentence, terms)
        if sentence_score <= 0:
            continue
        selected.append(sentence)
        total_length += len(sentence)
        if total_length >= limit:
            break
    return compact_text(" ".join(selected) if selected else ranked[0], limit=limit)


def split_sentences(text: str) -> list[str]:
    """Split filing text into compact sentence-like units."""

    cleaned = re.sub(r"\s+", " ", clean_filing_text(text)).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def compact_text(text: str, *, limit: int = 600) -> str:
    """Collapse whitespace and trim long passages."""

    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."
