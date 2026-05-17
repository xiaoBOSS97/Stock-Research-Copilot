"""Recent company news download and cache helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pandas as pd
import requests


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
DEFAULT_NEWS_DIR = Path("data/news")
NEWS_COLUMNS = [
    "published_at",
    "title",
    "source",
    "summary",
    "url",
    "overall_sentiment_label",
    "overall_sentiment_score",
]


class NewsDataError(RuntimeError):
    """Raised when recent news cannot be downloaded or parsed."""


def get_alpha_vantage_api_key() -> str | None:
    """Read the Alpha Vantage API key from the local environment."""

    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    return api_key.strip() if api_key else None


def fetch_alpha_vantage_news(
    ticker: str,
    *,
    limit: int = 20,
    days_back: int = 30,
    api_key: str | None = None,
    session: Any = requests,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch recent news from Alpha Vantage's news sentiment endpoint."""

    symbol = normalize_ticker(ticker)
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if days_back <= 0:
        raise ValueError("days_back must be positive.")

    resolved_api_key = api_key or get_alpha_vantage_api_key()
    if not resolved_api_key:
        raise NewsDataError("Missing ALPHA_VANTAGE_API_KEY in environment.")

    now = datetime.now(timezone.utc)
    time_from = now - timedelta(days=days_back)
    try:
        response = session.get(
            ALPHA_VANTAGE_URL,
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "time_from": time_from.strftime("%Y%m%dT%H%M"),
                "limit": str(limit),
                "apikey": resolved_api_key,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise NewsDataError(f"Could not download news: {exc}") from exc
    except ValueError as exc:
        raise NewsDataError("News provider returned invalid JSON.") from exc

    _raise_for_provider_error(payload)
    return alpha_vantage_news_payload_to_frame(payload, ticker=symbol, limit=limit)


def alpha_vantage_news_payload_to_frame(
    payload: dict[str, Any],
    *,
    ticker: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Convert an Alpha Vantage news response into a normalized DataFrame."""

    feed = payload.get("feed")
    if feed is None:
        feed = payload.get("Feed")
    if not isinstance(feed, list):
        return empty_news_frame()

    rows = []
    for item in feed:
        if not isinstance(item, dict):
            continue
        if ticker and not _article_mentions_ticker(item, ticker):
            continue
        rows.append(
            {
                "published_at": parse_alpha_vantage_time(item.get("time_published")),
                "title": _clean_text(item.get("title")),
                "source": _clean_text(item.get("source")),
                "summary": _clean_text(item.get("summary")),
                "url": _clean_text(item.get("url")),
                "overall_sentiment_label": _clean_text(item.get("overall_sentiment_label")),
                "overall_sentiment_score": _to_float(item.get("overall_sentiment_score")),
            }
        )

    frame = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    if frame.empty:
        return empty_news_frame()
    frame = frame.sort_values("published_at", ascending=False, na_position="last").reset_index(drop=True)
    if limit is not None:
        frame = frame.head(limit)
    return frame


def save_news_cache(
    ticker: str,
    news: pd.DataFrame,
    output_dir: str | Path = DEFAULT_NEWS_DIR,
) -> Path:
    """Save recent news to ``data/news/{ticker}_latest_news.json``."""

    symbol = normalize_ticker(ticker)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{symbol}_latest_news.json"
    serializable = news.copy()
    if "published_at" in serializable.columns:
        serializable["published_at"] = serializable["published_at"].map(_datetime_to_iso)
    output_path.write_text(
        json.dumps(serializable.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return output_path


def load_news_cache(ticker: str, input_dir: str | Path = DEFAULT_NEWS_DIR) -> pd.DataFrame:
    """Load cached recent news, returning an empty frame when missing."""

    symbol = normalize_ticker(ticker)
    path = Path(input_dir) / f"{symbol}_latest_news.json"
    if not path.exists():
        return empty_news_frame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NewsDataError(f"Cached news file is invalid JSON: {path}") from exc
    if not isinstance(payload, list):
        raise NewsDataError(f"Cached news file has unexpected shape: {path}")
    frame = pd.DataFrame(payload, columns=NEWS_COLUMNS)
    if frame.empty:
        return empty_news_frame()
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    frame["overall_sentiment_score"] = pd.to_numeric(
        frame["overall_sentiment_score"],
        errors="coerce",
    )
    return frame


def get_recent_news(
    ticker: str,
    *,
    limit: int = 20,
    days_back: int = 30,
    cache: bool = True,
    cache_dir: str | Path = DEFAULT_NEWS_DIR,
) -> pd.DataFrame:
    """Download recent ticker news and optionally cache the normalized result."""

    frame = fetch_alpha_vantage_news(ticker, limit=limit, days_back=days_back)
    if cache:
        save_news_cache(ticker, frame, output_dir=cache_dir)
    return frame


def summarize_news(news: pd.DataFrame, *, max_items: int = 5) -> str:
    """Build a compact report-ready news summary."""

    if news.empty:
        return ""

    rows = []
    for _, row in news.head(max_items).iterrows():
        date_text = _display_date(row.get("published_at"))
        source = row.get("source") or "Unknown source"
        title = row.get("title") or "Untitled article"
        sentiment = row.get("overall_sentiment_label") or "N/A"
        rows.append(f"- {date_text} | {source} | {sentiment}: {title}")
    return "\n".join(rows)


def empty_news_frame() -> pd.DataFrame:
    """Return an empty news frame with stable columns."""

    return pd.DataFrame(columns=NEWS_COLUMNS)


def normalize_ticker(ticker: str) -> str:
    """Normalize a ticker symbol for provider requests and cache names."""

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker must not be empty.")
    return symbol


def parse_alpha_vantage_time(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse Alpha Vantage timestamps such as ``20240507T143000``."""

    if not value:
        return pd.NaT
    timestamp = pd.to_datetime(str(value), format="%Y%m%dT%H%M%S", errors="coerce", utc=True)
    if pd.isna(timestamp):
        timestamp = pd.to_datetime(str(value), errors="coerce", utc=True)
    return timestamp


def _article_mentions_ticker(item: dict[str, Any], ticker: str) -> bool:
    ticker_sentiment = item.get("ticker_sentiment")
    if not ticker_sentiment:
        return True
    if not isinstance(ticker_sentiment, list):
        return True
    symbol = ticker.upper()
    return any(
        isinstance(entry, dict) and str(entry.get("ticker", "")).upper() == symbol
        for entry in ticker_sentiment
    )


def _raise_for_provider_error(payload: dict[str, Any]) -> None:
    for key in ("Error Message", "Information", "Note"):
        message = payload.get(key)
        if message:
            raise NewsDataError(str(message))


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _to_float(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _datetime_to_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    return timestamp.isoformat()


def _display_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return pd.Timestamp(value).date().isoformat()
