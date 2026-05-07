from __future__ import annotations

from src.utils.glossary import GLOSSARY, abbr, term_help


def test_term_help_returns_known_definition() -> None:
    assert "Relative Strength Index" in term_help("RSI")
    assert "Discounted cash flow" in term_help("DCF")
    assert "fluctuates" in term_help("Volatility")
    assert "year-over-year revenue growth" in term_help("Revenue Growth")


def test_abbr_adds_hover_title() -> None:
    html = abbr("RSI")

    assert html.startswith("<abbr")
    assert 'title="' in html
    assert ">RSI</abbr>" in html


def test_glossary_contains_core_terms() -> None:
    for term in ["RSI", "P/E", "P/S", "DCF", "Bear", "Base", "Bull", "Net Income Growth YoY"]:
        assert term in GLOSSARY
