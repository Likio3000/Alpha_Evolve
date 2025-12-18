#!/usr/bin/env bash
set -euo pipefail

# Minimal end-to-end smoke run on tiny data
# Usage: bash scripts/smoke_run.sh [--sp500]

here=$(cd "$(dirname "$0")" && pwd -P)
root="$here/.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-$root/.uv_cache}"

if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run python)
elif [ -x "$root/.venv/bin/python" ]; then
  RUNNER=("$root/.venv/bin/python")
elif command -v python3 >/dev/null 2>&1; then
  RUNNER=(python3)
else
  RUNNER=(python)
fi

if [[ "${1:-}" == "--sp500" ]]; then
  cfg="configs/sp500.toml"
  data_dir="data_sp500"
else
  cfg=""
  data_dir="tests/data/good"
fi

cd "$root"

if [[ -n "$cfg" ]]; then
  echo "Running pipeline with $cfg (MOEA+MF+Ensemble demo)"
  "${RUNNER[@]}" -m alpha_evolve.cli.pipeline 3 --config "$cfg" \
    --selection_metric phased --ic_phase_gens 1 \
    --moea_enabled --moea_elite_frac 0.25 \
    --mf_enabled --mf_initial_fraction 0.5 --mf_promote_fraction 0.5 --mf_min_promote 4 \
    --cv_k_folds 0 --cv_embargo 0 \
    --ensemble_mode --ensemble_size 2 --ensemble_max_corr 0.5 \
    --disable-align-cache --debug_prints
else
  echo "Running pipeline on tests/data/good (MOEA+MF+Ensemble demo)"
  "${RUNNER[@]}" -m alpha_evolve.cli.pipeline 3 --data_dir "$data_dir" --min_common_points 3 \
    --max_lookback_data_option full_overlap --selection_metric phased --ic_phase_gens 1 \
    --top_to_backtest 2 --fee 0.5 \
    --moea_enabled --moea_elite_frac 0.5 \
    --mf_enabled --mf_initial_fraction 0.5 --mf_promote_fraction 0.5 --mf_min_promote 2 \
    --cv_k_folds 0 --cv_embargo 0 \
    --ensemble_mode --ensemble_size 2 --ensemble_max_corr 0.5 \
    --disable-align-cache --debug_prints
fi

echo "Smoke run complete. See pipeline_runs_cs/LATEST."
