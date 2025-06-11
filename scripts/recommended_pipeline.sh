#!/bin/sh
# Example invocation of run_pipeline.py using parameters recommended by the
# AlphaEvolve paper with a few useful extras.
# Adjust --data_dir as needed.
uv run run_pipeline.py 10 \
  --max_lookback_data_option full_overlap \
  --pop_size 100 \
  --tournament_k 10 \
  --p_mut 0.9 \
  --p_cross 0.0 \
  --max_setup_ops 21 \
  --max_predict_ops 21 \
  --max_update_ops 45 \
  --max_scalar_operands 10 \
  --max_vector_operands 16 \
  --max_matrix_operands 4 \
  --eval_cache_size 128 \
  --corr_cutoff 0.15 \
  --fee 0.5 \
  --long_short_n 0 \
  --workers 4 \
  --run_baselines \
  --debug_prints
  # baseline metrics are cached; use --retrain_baselines to refresh

