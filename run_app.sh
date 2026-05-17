#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

HOST="${STREAMLIT_HOST:-0.0.0.0}"
PORT="${STREAMLIT_PORT:-8501}"

exec uv run streamlit run src/app/streamlit_app.py \
  --server.address "$HOST" \
  --server.port "$PORT"
