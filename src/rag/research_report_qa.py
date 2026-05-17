"""Local financial organization report analysis and search helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd


RESEARCH_REPORT_TOPICS: dict[str, tuple[str, ...]] = {
    "Rating / Recommendation": (
        "rating",
        "recommendation",
        "buy",
        "hold",
        "sell",
        "outperform",
        "underperform",
        "overweight",
        "underweight",
    ),
    "Price Target / Valuation": (
        "price target",
        "target price",
        "valuation",
        "multiple",
        "discounted cash flow",
        "dcf",
        "upside",
        "downside",
    ),
    "Growth Drivers": (
        "growth",
        "revenue",
        "demand",
        "market share",
        "product",
        "customer",
        "pipeline",
    ),
    "Margins / Profitability": (
        "margin",
        "profitability",
        "gross margin",
        "operating margin",
        "cost",
        "expense",
        "leverage",
    ),
    "Risks / Debates": (
        "risk",
        "downside",
        "bear case",
        "competition",
        "regulatory",
        "macro",
        "execution",
    ),
}


@dataclass(frozen=True)
class ResearchReportChunk:
    """A searchable chunk from an uploaded external research report."""

    chunk_id: int
    text: str


def clean_research_report_text(text: str) -> str:
    """Normalize report text for storage and retrieval."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_research_report(
    text: str,
    chunk_size: int = 1400,
    overlap: int = 180,
) -> list[ResearchReportChunk]:
    """Split research report text into overlapping chunks."""

    cleaned = clean_research_report_text(text)
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
        chunks.append(ResearchReportChunk(chunk_id=chunk_id, text=cleaned[start:end].strip()))
        if end == len(cleaned):
            break
        start = end - overlap
        chunk_id += 1
    return chunks


def search_research_report(
    text: str,
    query: str,
    top_k: int = 5,
    chunk_size: int = 1400,
    overlap: int = 180,
) -> pd.DataFrame:
    """Return top report passages for a plain-language query."""

    terms = query_terms(query)
    chunks = chunk_research_report(text, chunk_size=chunk_size, overlap=overlap)
    rows = []
    for chunk in chunks:
        score = score_text(chunk.text, terms)
        if score <= 0:
            continue
        rows.append(
            {
                "Chunk": chunk.chunk_id,
                "Score": score,
                "Passage": best_passage(chunk.text, terms),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Chunk", "Score", "Passage"])
    return pd.DataFrame(rows).sort_values(["Score", "Chunk"], ascending=[False, True]).head(top_k)


def analyze_research_report(text: str) -> dict[str, Any]:
    """Build a compact analysis pack for an uploaded external research report."""

    cleaned = clean_research_report_text(text)
    sentences = split_sentences(cleaned)
    topic_rows = []
    for topic, keywords in RESEARCH_REPORT_TOPICS.items():
        matches = matching_sentences(sentences, keywords)
        topic_rows.append(
            {
                "Topic": topic,
                "Mentions": len(matches),
                "Representative Excerpt": matches[0] if matches else "N/A",
            }
        )

    return {
        "word_count": len(re.findall(r"\b\w+\b", cleaned)),
        "chunk_count": len(chunk_research_report(cleaned)),
        "topic_table": pd.DataFrame(topic_rows),
        "summary": summarize_research_report(cleaned, topic_rows),
        "suggested_questions": suggested_questions(topic_rows),
    }


def summarize_research_report(text: str, topic_rows: list[dict[str, Any]] | None = None) -> str:
    """Create a short report-ready summary from extracted external research text."""

    cleaned = clean_research_report_text(text)
    if not cleaned:
        return ""
    rows = topic_rows
    if rows is None:
        rows = analyze_research_report(cleaned)["topic_table"].to_dict(orient="records")

    bullets = []
    for row in rows:
        excerpt = str(row.get("Representative Excerpt") or "")
        if excerpt and excerpt != "N/A":
            bullets.append(f"- {row.get('Topic')}: {excerpt}")
        if len(bullets) >= 4:
            break
    return "\n".join(bullets)


def query_terms(query: str) -> list[str]:
    """Extract useful retrieval terms from a report question."""

    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query.lower())
    stopwords = {
        "about",
        "also",
        "and",
        "are",
        "company",
        "could",
        "does",
        "explain",
        "for",
        "from",
        "has",
        "have",
        "how",
        "report",
        "research",
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
    terms = []
    for term in raw_terms:
        singular = term[:-1] if term.endswith("s") and len(term) > 4 else term
        if singular not in stopwords and singular not in terms:
            terms.append(singular)
    return terms


def score_text(text: str, terms: list[str]) -> int:
    """Score text by simple term frequency."""

    lower = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(term)}\w*\b", lower)) for term in terms)


def best_passage(text: str, terms: list[str], *, limit: int = 600) -> str:
    """Return the strongest compact passage from a report chunk."""

    sentences = split_sentences(text)
    if not sentences:
        return compact_text(text, limit=limit)
    ranked = sorted(sentences, key=lambda sentence: score_text(sentence, terms), reverse=True)
    selected = []
    for sentence in ranked:
        if score_text(sentence, terms) <= 0:
            continue
        selected.append(sentence)
        if len(" ".join(selected)) >= limit * 0.75:
            break
    return compact_text(" ".join(selected) if selected else text, limit=limit)


def split_sentences(text: str) -> list[str]:
    """Split text into rough sentences for excerpts."""

    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    return [part.strip() for part in parts if part.strip()]


def matching_sentences(sentences: list[str], keywords: tuple[str, ...]) -> list[str]:
    """Return compact sentences that mention any supplied keyword."""

    matches = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            matches.append(compact_text(sentence, limit=280))
    return matches


def compact_text(text: str, limit: int = 600) -> str:
    """Compact whitespace and truncate long passages."""

    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."


def suggested_questions(topic_rows: list[dict[str, Any]]) -> list[str]:
    """Suggest useful questions based on detected report topics."""

    suggestions = [
        "What is the analyst's valuation argument?",
        "What are the main upside and downside drivers?",
    ]
    topics_with_mentions = {row["Topic"] for row in topic_rows if int(row.get("Mentions") or 0) > 0}
    if "Rating / Recommendation" in topics_with_mentions:
        suggestions.append("What rating or recommendation does the report discuss?")
    if "Risks / Debates" in topics_with_mentions:
        suggestions.append("What risks or debates does the report highlight?")
    return suggestions
