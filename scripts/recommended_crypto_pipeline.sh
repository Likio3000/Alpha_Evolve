#!/usr/bin/env bash
# Alpha Evolve – Recommended Crypto pipeline (4h bars)
#
# Usage:
#   bash scripts/recommended_crypto_pipeline.sh [extra-args]
#
# Env overrides:
#   GENS     – generations to evolve (default: 60)
#   DATA_DIR – data directory (default: data)
#   LOOKBACK – 'common_1200' (default) or 'full_overlap'
#   WORKERS  – parallel workers (default: 8)
#
# Notes:
# - 4h crypto bars → annualization factor 365*6.
# - Includes per-asset stop, moderate novelty penalties, and drawdown limiter.
# - Vol targeting optional; crypto can be more volatile than SP500.

set -eu

GENS="${GENS:-60}"
DATA_DIR="${DATA_DIR:-data}"
LOOKBACK="${LOOKBACK:-common_1200}"
WORKERS="${WORKERS:-8}"
ANNUAL=$((365*6))

echo "[recommended_crypto] gens=${GENS} data=${DATA_DIR} lookback=${LOOKBACK} workers=${WORKERS} annual=${ANNUAL}" >&2

ARGS=(
  run_pipeline.py
  "${GENS}"
  --data_dir "${DATA_DIR}"
  --max_lookback_data_option "${LOOKBACK}"
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
  --dd_limit 0.20
  --dd_reduction 0.4
  --log-level INFO
)

if command -v uv >/dev/null 2>&1; then
  uv run "${ARGS[@]}" "$@"
else
  python "${ARGS[@]}" "$@"
fi

echo "[recommended_crypto] Done." >&2
