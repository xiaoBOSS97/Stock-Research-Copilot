"""SEC filing text extraction, section parsing, and chunking helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_PARSED_FILING_DIR = Path("data/processed/filings")
SUPPORTED_FILING_SUFFIXES = {".htm", ".html", ".txt"}

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "Business": ("1",),
    "Risk Factors": ("1A",),
    "Unresolved Staff Comments": ("1B",),
    "Properties": ("2",),
    "Legal Proceedings": ("3",),
    "Management Discussion and Analysis": ("7",),
    "Market Risk": ("7A",),
    "Financial Statements": ("8",),
}


@dataclass(frozen=True)
class FilingSection:
    """A source-traceable section extracted from an SEC filing."""

    section: str
    item: str | None
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a serializable representation of the section."""

        return asdict(self)


@dataclass(frozen=True)
class FilingChunk:
    """A retrieval-ready chunk from one filing section."""

    chunk_id: int
    section: str
    item: str | None
    text: str
    source_ref: str

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a serializable representation of the chunk."""

        return asdict(self)


class _FilingHTMLTextExtractor(HTMLParser):
    """Small HTML-to-text extractor tuned for SEC filing documents."""

    _BLOCK_TAGS = {
        "address",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(data)

    def text(self) -> str:
        """Return extracted text."""

        return "".join(self._parts)


def load_filing_text(path: str | Path) -> str:
    """Load a saved SEC filing and return clean plain text."""

    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_FILING_SUFFIXES:
        raise ValueError("SEC filing files must be .htm, .html, or .txt.")
    raw = source.read_text(encoding="utf-8", errors="ignore")
    if source.suffix.lower() in {".htm", ".html"} or _looks_like_html(raw):
        return html_to_text(raw)
    return clean_filing_text(raw)


def html_to_text(html: str) -> str:
    """Convert SEC filing HTML into normalized plain text."""

    extractor = _FilingHTMLTextExtractor()
    extractor.feed(html)
    extractor.close()
    return clean_filing_text(unescape(extractor.text()))


def clean_filing_text(text: str) -> str:
    """Normalize whitespace while preserving line breaks useful for item headings."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_filing_sections(text: str) -> list[FilingSection]:
    """Extract major 10-K/10-Q sections from clean filing text."""

    cleaned = clean_filing_text(text)
    if not cleaned:
        return []

    headings = _find_item_headings(cleaned)
    if len(headings) < 2:
        return [FilingSection("Filing Text", None, cleaned, 0, len(cleaned))]

    sections: list[FilingSection] = []
    for index, heading in enumerate(headings):
        item, start = heading
        end = headings[index + 1][1] if index + 1 < len(headings) else len(cleaned)
        label = _section_label_for_item(item)
        if label is None:
            continue
        section_text = _trim_section_heading(cleaned[start:end])
        if section_text:
            sections.append(
                FilingSection(
                    section=label,
                    item=item,
                    text=section_text,
                    start_char=start,
                    end_char=end,
                )
            )

    if not sections:
        return [FilingSection("Filing Text", None, cleaned, 0, len(cleaned))]
    return sections


def parse_filing_file(path: str | Path) -> list[FilingSection]:
    """Load a filing from disk and extract source sections."""

    return extract_filing_sections(load_filing_text(path))


def chunk_filing_sections(
    sections: list[FilingSection],
    *,
    ticker: str | None = None,
    filing_type: str | None = None,
    filing_date: str | None = None,
    chunk_size: int = 1800,
    overlap: int = 200,
) -> list[FilingChunk]:
    """Split filing sections into overlapping retrieval-ready chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[FilingChunk] = []
    chunk_id = 1
    source_prefix = " | ".join(part for part in (ticker, filing_type, filing_date) if part)
    for section in sections:
        start = 0
        text = clean_filing_text(section.text)
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                source_parts = [source_prefix, section.section]
                if section.item:
                    source_parts.append(f"Item {section.item}")
                source_ref = " | ".join(part for part in source_parts if part)
                chunks.append(
                    FilingChunk(
                        chunk_id=chunk_id,
                        section=section.section,
                        item=section.item,
                        text=chunk_text,
                        source_ref=source_ref,
                    )
                )
                chunk_id += 1
            if end == len(text):
                break
            start = end - overlap
    return chunks


def save_parsed_filing(
    sections: list[FilingSection],
    *,
    ticker: str,
    accession_number: str,
    output_dir: str | Path = DEFAULT_PARSED_FILING_DIR,
) -> Path:
    """Save parsed filing sections as JSON for later RAG indexing."""

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker must not be empty.")
    safe_accession = re.sub(r"[^A-Za-z0-9_.-]+", "_", accession_number).strip("._")
    target_dir = Path(output_dir) / symbol
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_accession}_sections.json"
    payload: dict[str, Any] = {
        "ticker": symbol,
        "accession_number": accession_number,
        "sections": [section.to_dict() for section in sections],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _find_item_headings(text: str) -> list[tuple[str, int]]:
    pattern = re.compile(
        r"(?im)^\s*item\s+"
        r"(?P<item>1A|1B|7A|[1-9])"
        r"\.?\s*(?:[-—:.\s]+[A-Z][^\n]{0,120})?$"
    )
    matches = [(match.group("item").upper(), match.start()) for match in pattern.finditer(text)]
    return _dedupe_heading_matches(matches)


def _dedupe_heading_matches(matches: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Remove table-of-contents heading runs before the real filing body."""

    if not matches:
        return []

    deduped: list[tuple[str, int]] = []
    last_item: str | None = None
    for item, position in matches:
        if item == last_item and deduped and position - deduped[-1][1] < 500:
            continue
        deduped.append((item, position))
        last_item = item

    item_positions: dict[str, list[int]] = {}
    for item, position in deduped:
        item_positions.setdefault(item, []).append(position)

    first_repeated_position = min(
        positions[1]
        for positions in item_positions.values()
        if len(positions) > 1
    ) if any(len(positions) > 1 for positions in item_positions.values()) else None
    if first_repeated_position is None:
        return deduped
    return [(item, position) for item, position in deduped if position >= first_repeated_position]


def _section_label_for_item(item: str) -> str | None:
    for label, aliases in SECTION_ALIASES.items():
        if item in aliases:
            return label
    return None


def _trim_section_heading(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    if lines and re.match(r"(?i)^item\s+(1A|1B|7A|[1-9])\b", lines[0]):
        lines.pop(0)
    return clean_filing_text("\n".join(lines))


def _looks_like_html(text: str) -> bool:
    return bool(re.search(r"(?is)<\s*(html|body|document|ix:|div|table|p)\b", text))
