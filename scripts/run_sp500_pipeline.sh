#!/usr/bin/env bash
# Run the Alpha Evolve pipeline using the S&P 500 daily dataset.
#
# Usage:
#   bash scripts/run_sp500_pipeline.sh [extra-args]
#
# Env overrides:
#   GENS     – generations to evolve (default: 5)
#   DATA_DIR – data directory (default: data_sp500)
#   LOOKBACK – max_lookback_data_option (default: common_1200)
#   ANNUAL   – annualization factor for daily bars (default: 252)
#
# Examples:
#   bash scripts/run_sp500_pipeline.sh
#   GENS=10 LOOKBACK=full_overlap bash scripts/run_sp500_pipeline.sh --run_baselines

set -eu

GENS="${GENS:-5}"
DATA_DIR="${DATA_DIR:-data_sp500}"
# Use maximum compatible overlap across all symbols by default for daily SP500
LOOKBACK="${LOOKBACK:-full_overlap}"
ANNUAL="${ANNUAL:-252}"

echo "[run_sp500_pipeline] gens=${GENS} data_dir=${DATA_DIR} lookback=${LOOKBACK} annual=${ANNUAL}" >&2

if command -v uv >/dev/null 2>&1; then
  uv run python -m alpha_evolve.cli.pipeline "${GENS}" \
    --data_dir "${DATA_DIR}" \
    --max_lookback_data_option "${LOOKBACK}" \
    --annualization_factor "${ANNUAL}" \
    --selection_metric phased --ic_phase_gens 5 \
    --moea_enabled --moea_elite_frac 0.25 \
    --cv_k_folds 4 --cv_embargo 5 \
    --hof_corr_mode per_bar \
    --ensemble_mode --ensemble_size 5 --ensemble_max_corr 0.3 \
    "$@"
else
  python3 -m alpha_evolve.cli.pipeline "${GENS}" \
    --data_dir "${DATA_DIR}" \
    --max_lookback_data_option "${LOOKBACK}" \
    --annualization_factor "${ANNUAL}" \
    --selection_metric phased --ic_phase_gens 5 \
    --moea_enabled --moea_elite_frac 0.25 \
    --cv_k_folds 4 --cv_embargo 5 \
    --hof_corr_mode per_bar \
    --ensemble_mode --ensemble_size 5 --ensemble_max_corr 0.3 \
    "$@"
fi

echo "[run_sp500_pipeline] Done." >&2
