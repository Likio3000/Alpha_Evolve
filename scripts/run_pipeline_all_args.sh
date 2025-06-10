uv run run_pipeline.py 15 \
  --seed 42 \
  --pop_size 100 \
  --tournament_k 10 \
  --p_mut 0.9 \
  --p_cross 0.4 \
  --elite_keep 6 \
  --fresh_rate 0.25 \
  --max_ops 87 \
  --max_setup_ops 21 \
  --max_predict_ops 21 \
  --max_update_ops 45 \
  --max_scalar_operands 10 \
  --max_vector_operands 16 \
  --max_matrix_operands 4 \
  --parsimony_penalty 0.002 \
  --corr_penalty_w 0.25 \
  --corr_cutoff 0.15 \
  --xs_flat_guard 0.02 \
  --t_flat_guard 0.005 \
  --early_abort_bars 60 \
  --early_abort_xs 0.02 \
  --early_abort_t 0.005 \
  --hof_size 20 \
  --scale zscore \
  --workers 2 \
  --eval_cache_size 128 \
  --data_dir ./data \
  --max_lookback_data_option common_1200 \
  --min_common_points 1200 \
  --eval_lag 1 \
  --top 10 \
  --fee 1.0 \
  --hold 1 \
  --debug_prints \
  --run_baselines \
  # baseline metrics are cached; use --retrain_baselines to refresh
  # --log-level DEBUG
  # --log-file
  # --quiet
  # --annualization_factor (int)
