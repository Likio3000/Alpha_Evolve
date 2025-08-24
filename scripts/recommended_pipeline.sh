#!/bin/sh
# Alpha Evolve – Recommended pipeline (uv run)
#
# Quick start:
#   - Ensure uv is installed: https://docs.astral.sh/uv/
#   - From the repo root, run:  sh scripts/recommended_pipeline.sh
#   - Adjust --data_dir if your dataset is not in ./data
#
# Notes:
#   - Uses stronger exploration, tighter novelty, early-abort guards,
#     and mid-risk penalties as discussed.
#   - Logging defaults to INFO; pass --debug_prints if you want verbose traces.

uv run run_pipeline.py 100 \
  --pop_size 240 \
  --workers 12 \
  --use_train_val_splits \
  --train_points 840 \
  --val_points 360 \
  --scale winsor \
  --winsor_p 0.02 \
  --corr_cutoff 0.10 \
  --corr_penalty_w 0.40 \
  --ic_std_penalty_w 0.03 \
  --turnover_penalty_w 0.01 \
  --parsimony_penalty 0.0035 \
  --parsimony_jitter_pct 0.10 \
  --p_mut 0.90 \
  --fresh_rate 0.30 \
  --tournament_k 3 \
  --ops_split_jitter 0.25 \
  --hof_per_gen 3 \
  --early_abort_bars 12 \
  --early_abort_xs 0.07 \
  --early_abort_t 0.07 \
  --flat_bar_threshold 0.35 \
  --stop_loss_pct 0.03 \
  --long_short_n 2 \
  --hold 2 \
  --bt-scale winsor \
  # Risk controls (backtest):
  # Drawdown limiter (keep; you can tune):
  --dd_limit 0.15 \
  --dd_reduction 0.3 \
  # Optional risk controls you may enable later:
  # --sector_neutralize_positions \
  # --volatility_target 0.01 --volatility_lookback 30 \
  # --min_leverage 0.25 --max_leverage 2.0 \
  --log-level INFO
  # Optional extras:
  #   --run_baselines           # also train GA tree + RankLSTM baselines
  #   --retrain_baselines       # ignore cached baseline metrics and retrain
  #   --data_dir /path/to/data  # if your data is not under ./data
