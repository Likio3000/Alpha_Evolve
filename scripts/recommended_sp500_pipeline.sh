#!/usr/bin/env bash
# Alpha Evolve – Recommended SP500 pipeline (daily bars)
#
# Usage:
#   bash scripts/recommended_sp500_pipeline.sh [extra-args]
#
# Env overrides:
#   GENS     – generations to evolve (default: 60)
#   DATA_DIR – data directory (default: data_sp500)
#   LOOKBACK – 'full_overlap' (default, aligns on max common) or 'common_1200'
#   MIN_POINTS – minimum common eval points (default: 1200)
#   WORKERS  – parallel workers (default: 8)
#
# Notes:
# - Daily bars → annualization factor 252.
# - Includes per-asset stop, moderate novelty penalties, and drawdown limiter.
# - Vol targeting is optional (commented); enable for tighter DD control.

set -eu

GENS="${GENS:-60}"
DATA_DIR="${DATA_DIR:-data_sp500}"
LOOKBACK="${LOOKBACK:-full_overlap}"
MIN_POINTS="${MIN_POINTS:-1200}"
WORKERS="${WORKERS:-8}"
ANNUAL=252

echo "[recommended_sp500] gens=${GENS} data=${DATA_DIR} lookback=${LOOKBACK} min_points=${MIN_POINTS} workers=${WORKERS} annual=${ANNUAL}" >&2

ARGS=(
  run_pipeline.py
  "${GENS}"
  --data_dir "${DATA_DIR}"
  --max_lookback_data_option "${LOOKBACK}"
  --min_common_points "${MIN_POINTS}"
  --annualization_factor "${ANNUAL}"
  --pop_size 200
  --workers "${WORKERS}"
  --use_train_val_splits
  --train_points 840
  --val_points 360
  --scale winsor
  --winsor_p 0.02
  --corr_cutoff 0.10
  --corr_penalty_w 0.40
  --ic_std_penalty_w 0.03
  --turnover_penalty_w 0.01
  --parsimony_penalty 0.0035
  --parsimony_jitter_pct 0.10
  --p_mut 0.90
  --p_cross 0.35
  --fresh_rate 0.25
  --tournament_k 3
  --ops_split_jitter 0.25
  --hof_per_gen 3
  --early_abort_bars 12
  --early_abort_xs 0.07
  --early_abort_t 0.07
  --flat_bar_threshold 0.35
  --stop_loss_pct 0.03
  --hold 2
  --long_short_n 2
  --bt-scale winsor
  --dd_limit 0.15
  --dd_reduction 0.3
  --log-level INFO
)

if command -v uv >/dev/null 2>&1; then
  uv run "${ARGS[@]}" "$@"
else
  python "${ARGS[@]}" "$@"
fi

echo "[recommended_sp500] Done." >&2
