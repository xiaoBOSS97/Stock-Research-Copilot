from __future__ import annotations

from src.utils.formatting import format_large_number, format_value, humanize_label


def test_humanize_label_uses_overrides() -> None:
    assert humanize_label("net_margin") == "Net Margin"
    assert humanize_label("fcf_margin") == "FCF Margin"
    assert humanize_label("pe_target_price") == "P/E Target Price"
    assert humanize_label("dcf_target_price") == "DCF Target Price"
    assert humanize_label("revenue_cagr") == "Revenue CAGR"
    assert humanize_label("implied_growth_rate") == "Implied FCF Growth"


def test_format_value_handles_percent_and_large_numbers() -> None:
    assert format_value(0.251, "net_margin") == "25.1%"
    assert format_value(4_200_000_000_000, "marketCap") == "4.20T"
    assert format_value(100_000_000_000, "base_free_cash_flow") == "100.00B"
    assert format_value(0.1234, "implied_growth_rate") == "12.3%"
    assert format_value(None, "revenue") == "N/A"


def test_format_large_number_uses_suffixes() -> None:
    assert format_large_number(1_500_000_000) == "1.50B"
    assert format_large_number(2_000_000) == "2.00M"
