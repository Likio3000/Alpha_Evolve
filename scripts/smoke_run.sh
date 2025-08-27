#!/usr/bin/env bash
set -euo pipefail

# Minimal end-to-end smoke run on tiny data
# Usage: bash scripts/smoke_run.sh [--sp500]

here=$(cd "$(dirname "$0")" && pwd -P)
root="$here/.."

if [[ "${1:-}" == "--sp500" ]]; then
  cfg="configs/sp500.toml"
  data_dir="data_sp500"
else
  cfg=""
  data_dir="tests/data/good"
fi

cd "$root"

if [[ -n "$cfg" ]]; then
  echo "Running pipeline with $cfg"
  uv run run_pipeline.py 3 --config "$cfg" --selection_metric auto --disable-align-cache --debug_prints
else
  echo "Running pipeline on tests/data/good"
  uv run run_pipeline.py 3 --data_dir "$data_dir" --min_common_points 10 \
    --max_lookback_data_option full_overlap --selection_metric auto \
    --top_to_backtest 2 --fee 0.5 --disable-align-cache --debug_prints
fi

echo "Smoke run complete. See pipeline_runs_cs/LATEST."

