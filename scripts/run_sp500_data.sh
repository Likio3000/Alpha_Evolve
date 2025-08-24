#!/usr/bin/env bash
# Run S&P 500 data fetch with defaults.
#
# Usage:
#   bash scripts/run_sp500_data.sh [extra-args]
#
# Env overrides:
#   OUT_DIR (default: data_sp500)
#   YEARS   (default: 20)
#   PAUSE   (default: 0.2)
#
# Examples:
#   bash scripts/run_sp500_data.sh
#   OUT_DIR=data_sp500_yearly YEARS=10 bash scripts/run_sp500_data.sh
#   bash scripts/run_sp500_data.sh --no-auto-adjust

set -eu

OUT_DIR="${OUT_DIR:-data_sp500}"
YEARS="${YEARS:-20}"
PAUSE="${PAUSE:-0.2}"

echo "[run_sp500_data] Fetching S&P 500 daily OHLC -> ${OUT_DIR} (years=${YEARS}, pause=${PAUSE})" >&2

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/fetch_sp500_data.py --out "${OUT_DIR}" --years "${YEARS}" --pause "${PAUSE}" "$@"
else
  python scripts/fetch_sp500_data.py --out "${OUT_DIR}" --years "${YEARS}" --pause "${PAUSE}" "$@"
fi

echo "[run_sp500_data] Done." >&2
