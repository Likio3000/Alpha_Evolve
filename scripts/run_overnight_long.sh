#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd -P)"
ts=$(date +"%Y%m%d_%H%M%S")
log_base="$root/logs/overnight_${ts}"

mkdir -p "$root/logs"
export PYTHONPATH="$root/src"

# Redirect stdout/stderr for launchd runs.
exec >"${log_base}.out" 2>&1

exec "$root/.venv312/bin/python" -m alpha_evolve.cli.pipeline 120 \
  --config "$root/configs/bench_sp500_full.toml" \
  --pop_size 240 \
  --eval_lag 2 \
  --hold 2 \
  --scale rank \
  --bt-scale rank \
  --selection_metric sharpe \
  --sharpe_proxy_w 1.0 \
  --corr_penalty_w 0.0 \
  --ic_std_penalty_w 0.0 \
  --turnover_penalty_w 0.0 \
  --relation_ops_weight 4.0 \
  --net_exposure_target 0.25 \
  --volatility_target 0.012 \
  --max_leverage 3.0 \
  --dd_limit 0.15 \
  --dd_reduction 0.5 \
  --seed 101 \
  --workers 1 \
  --output-dir "$root/pipeline_runs_cs" \
  --log-file "${log_base}.log"
