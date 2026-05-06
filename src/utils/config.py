"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_settings(path: str | Path = PROJECT_ROOT / "config" / "settings.yaml") -> dict[str, Any]:
    """Load YAML settings after reading local environment variables."""

    load_dotenv()
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    with settings_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
