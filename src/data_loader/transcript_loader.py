"""Earnings call transcript download helpers."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
import requests

from src.rag.transcript_analysis import clean_transcript_text, save_transcript_text


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class TranscriptDownloadError(RuntimeError):
    """Raised when a transcript cannot be downloaded or parsed."""


def normalize_quarter(quarter: str) -> str:
    """Normalize and validate fiscal quarter labels like ``2024Q1``."""

    cleaned = quarter.strip().upper().replace(" ", "")
    if not re.fullmatch(r"20\d{2}Q[1-4]", cleaned):
        raise ValueError("Quarter must use YYYYQ# format, for example 2024Q1.")
    return cleaned


def get_alpha_vantage_api_key() -> str | None:
    """Read the Alpha Vantage API key from the local environment."""

    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    return api_key.strip() if api_key else None


def download_alpha_vantage_transcript(
    ticker: str,
    quarter: str,
    *,
    api_key: str | None = None,
    session: Any = requests,
    timeout: int = 30,
) -> str:
    """Download an earnings call transcript from Alpha Vantage."""

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker must not be empty.")
    normalized_quarter = normalize_quarter(quarter)
    resolved_api_key = api_key or get_alpha_vantage_api_key()
    if not resolved_api_key:
        raise TranscriptDownloadError("Missing ALPHA_VANTAGE_API_KEY in environment.")

    try:
        response = session.get(
            ALPHA_VANTAGE_URL,
            params={
                "function": "EARNINGS_CALL_TRANSCRIPT",
                "symbol": symbol,
                "quarter": normalized_quarter,
                "apikey": resolved_api_key,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise TranscriptDownloadError(f"Could not download transcript: {exc}") from exc
    except ValueError as exc:
        raise TranscriptDownloadError("Transcript provider returned invalid JSON.") from exc

    _raise_for_provider_error(payload)
    transcript = alpha_vantage_payload_to_text(payload)
    if not transcript:
        raise TranscriptDownloadError(f"No transcript text returned for {symbol} {normalized_quarter}.")
    return transcript


def download_and_save_alpha_vantage_transcript(
    ticker: str,
    quarter: str,
    *,
    api_key: str | None = None,
    output_dir: str | Path = "data/transcripts",
) -> Path:
    """Download an Alpha Vantage transcript and save it locally."""

    symbol = ticker.strip().upper()
    normalized_quarter = normalize_quarter(quarter)
    transcript = download_alpha_vantage_transcript(symbol, normalized_quarter, api_key=api_key)
    return save_transcript_text(
        symbol,
        f"{symbol}_{normalized_quarter}_earnings_call.txt",
        transcript,
        output_dir=output_dir,
    )


def alpha_vantage_payload_to_text(payload: dict[str, Any]) -> str:
    """Convert Alpha Vantage transcript JSON into readable plain text."""

    transcript = payload.get("transcript")
    if transcript is None:
        transcript = payload.get("Transcript")

    if isinstance(transcript, str):
        return clean_transcript_text(transcript)
    if isinstance(transcript, list):
        parts = [_transcript_item_to_text(item) for item in transcript]
        return clean_transcript_text("\n\n".join(part for part in parts if part))
    if isinstance(transcript, dict):
        if isinstance(transcript.get("text"), str):
            return clean_transcript_text(transcript["text"])
        paragraphs = transcript.get("paragraphs")
        if isinstance(paragraphs, list):
            parts = [_transcript_item_to_text(item) for item in paragraphs]
            return clean_transcript_text("\n\n".join(part for part in parts if part))

    text_fields = [
        value
        for key, value in payload.items()
        if key.lower() in {"content", "text", "full_transcript"} and isinstance(value, str)
    ]
    return clean_transcript_text("\n\n".join(text_fields))


def _transcript_item_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""

    speaker = item.get("speaker") or item.get("name") or item.get("title")
    content = item.get("content") or item.get("text") or item.get("transcript")
    if not isinstance(content, str):
        return ""
    if speaker:
        return f"{speaker}: {content}"
    return content


def _raise_for_provider_error(payload: dict[str, Any]) -> None:
    for key in ("Error Message", "Information", "Note"):
        message = payload.get(key)
        if message:
            raise TranscriptDownloadError(str(message))
