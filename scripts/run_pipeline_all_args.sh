#!/bin/sh
# Example invocation of run_pipeline.py with every available CLI parameter
# explicitly set to its default value. Edit any value below to customise the
# evolution or back-test behaviour.

uv run run_pipeline.py 10 \
  --seed 42 \
  --pop_size 100 \
  --tournament_k 10 \
  --p_mut 0.9 \
  --p_cross 0.0 \
  --elite_keep 1 \
  --fresh_rate 0.12 \
  --max_ops 87 \
  --max_setup_ops 21 \
  --max_predict_ops 21 \
  --max_update_ops 45 \
  --max_scalar_operands 10 \
  --max_vector_operands 16 \
  --max_matrix_operands 4 \
  --parsimony_penalty 0.0001 \
  --corr_penalty_w 0.35 \
  --corr_cutoff 0.15 \
  # --keep_dupes_in_hof \  # enable to keep duplicates in the Hall of Fame
  --xs_flat_guard 0.05 \
  --t_flat_guard 0.05 \
  --early_abort_bars 20 \
  --early_abort_xs 0.05 \
  --early_abort_t 0.05 \
  --hof_size 20 \
  --scale zscore \
  # --quiet \  # uncomment to suppress most logging output
  --workers 1 \
  --eval_cache_size 128 \
  --data_dir ./data \
  --max_lookback_data_option common_1200 \
  --min_common_points 1200 \
  --eval_lag 1 \
  --top 10 \
  --fee 1.0 \
  --hold 1 \
  --annualization_factor 1512 \
  # --debug_prints \  # forward verbose output to the back-tester
  # --run_baselines \  # additionally train baseline models
  --log-level INFO \
  --log-file ""
