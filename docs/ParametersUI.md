UI Parameter Reference
======================

This document summarizes the parameters exposed in the Iterative Dashboard and
their meaning. Defaults mirror `config.EvolutionConfig`/`BacktestConfig`.

Core Evolution
- generations: Number of evolutionary generations.
- pop_size: Population size per generation.
- tournament_k: Contestants per tournament for parent selection.
- elite_keep: Top programs copied unchanged to next generation.
- hof_size: Maximum Hall of Fame entries.
- hof_per_gen: Top candidates added to HOF each generation (subject to filters).
- seed: Random seed for reproducibility.
- fresh_rate: Fraction of population replaced with brand new programs.
- p_mut, p_cross: Mutation and crossover probabilities.

Selection & Ramping
- selection_metric: Parent selection criterion: ramped fitness, fixed final weights, pure IC, auto-switch, or phased with IC warmup.
- ramp_fraction: Portion of total generations to reach full weights/penalties.
- ramp_min_gens: Minimum generations for the ramp.
- ic_phase_gens: Early IC-only generation count for the phased schedule.
- rank_softmax_beta_floor/target: Temperatures for rank-based tournament sampling.

Novelty & Correlation
- novelty_boost_w: Diversity boost for low correlation vs HOF predictions.
- novelty_struct_w: Structural novelty boost (opcode-set distance) vs HOF.
- hof_corr_mode: Correlation vs HOF computed on flattened series or per-bar.
- corr_penalty_w: Penalty applied to correlation vs HOF.
- corr_cutoff: Correlation threshold to drop near-duplicates vs HOF.

Fitness Weights
- sharpe_proxy_w: Weight for a Sharpe-like proxy in fitness.
- ic_std_penalty_w: Penalty for volatile IC.
- turnover_penalty_w: Penalty for high trading turnover.
- ic_tstat_w: Weight for IC t-statistic to reward stability.
- temporal_decay_half_life: Exponential half-life (bars) to weight recent data more.
- factor_penalty_w: Penalty weight applied to style-factor exposures (e.g., market, volatility, liquidity).
- factor_penalty_factors: Comma-separated factor names to neutralize (matches evaluation feature names such as `ret1d_t`).
- stress_penalty_w: Weight for the stress/transaction-cost robustness penalty.
- stress_fee_bps / stress_slippage_bps: Additional fees and slippage (in bps) used in the stress fitness.
- stress_shock_scale: Multiplier applied to downside returns when computing the stress fitness.

Evaluation Windows
- evaluation_horizons: Comma-separated prediction horizons (in bars) evaluated each generation.
- use_train_val_splits: Enable train/validation splits for evaluation metrics.
- train_points / val_points: Number of evaluation bars allocated to the train and validation slices.

Multi‑Objective & Fidelity
- moea_enabled: Enable Pareto selection (NSGA-II-like).
- moea_elite_frac: Portion of next generation chosen from the first Pareto front.
- mf_enabled: Enable multi-fidelity (cheap first pass then full re-eval for top‑K).
- mf_initial_fraction: Fraction of bars for the cheap first pass.
- mf_promote_fraction: Fraction of population promoted to full evaluation.
- mf_min_promote: Minimum number promoted regardless of fraction.

Quality Diversity Archive
- qd_archive_enabled: Maintain a MAP-Elites style archive alongside the Hall of Fame.
- qd_turnover_bins / qd_complexity_bins: Comma-separated descriptor bin edges for turnover and complexity.
- qd_max_entries: Maximum number of archive cells to keep.

Cross‑Validation
- cv_k_folds: K>1 enables CPCV-like purged CV.
- cv_embargo: Bars embargoed around each validation fold to reduce leakage.

Backtest Ensemble
- ensemble_mode: Also backtest an ensemble of top alphas.
- ensemble_size: Number of alphas to include (0 disables).
- ensemble_max_corr: Target max pairwise correlation when selecting ensemble members.

Evolution/Geneation Bias (EvolutionParams)
- vector_ops_bias: Probability to force vector-output ops when sampling.
- relation_ops_weight: Weight multiplier for relation_* ops.
- cs_ops_weight: Weight multiplier for cs_* ops.
- default_op_weight: Baseline weight for all ops.
- ops_split_base_setup/predict/update: Base split of ops across blocks.
- ops_split_jitter: Randomness added to the block split for fresh programs.

Data & Run
- data_dir: Directory with input CSVs.
- bt_top: How many evolved alphas to backtest and summarize.
- no_clean: Keep previous pipeline outputs.
- dry_run: Print resolved configs and planned outputs then exit.
- out_summary: Write SUMMARY.json with key artefacts for UI consumption.
