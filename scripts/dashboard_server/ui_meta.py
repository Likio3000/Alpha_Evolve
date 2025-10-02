from __future__ import annotations

from typing import Any, Dict
from django.http import JsonResponse


def get_evolution_params_ui_meta(request) -> JsonResponse:
    return JsonResponse({
        "schema_version": 1,
        "groups": [
            {
                "title": "Operator Biasing",
                "items": [
                    {"key": "vector_ops_bias", "label": "Vector Ops Bias", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Probability to force vector-output ops when sampling."},
                    {"key": "relation_ops_weight", "label": "Relation Ops Weight", "type": "float", "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5, "help": "Weight multiplier for relation_* ops during selection."},
                    {"key": "cs_ops_weight", "label": "Cross-sectional Ops Weight", "type": "float", "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.5, "help": "Weight multiplier for cs_* ops during selection."},
                    {"key": "default_op_weight", "label": "Default Op Weight", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "help": "Baseline weight for all ops."},
                ],
            },
            {
                "title": "Block Split",
                "items": [
                    {"key": "ops_split_base_setup", "label": "Setup Fraction", "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Base fraction of ops allocated to setup."},
                    {"key": "ops_split_base_predict", "label": "Predict Fraction", "type": "float", "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Base fraction of ops allocated to predict."},
                    {"key": "ops_split_base_update", "label": "Update Fraction", "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Base fraction of ops allocated to update."},
                    {"key": "ops_split_jitter", "label": "Split Jitter", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Randomness added to the block split when seeding fresh programs."},
                ],
            },
        ],
    })


def get_pipeline_params_ui_meta(request) -> JsonResponse:
    return JsonResponse({
        "schema_version": 1,
        "groups": [
            {
                "title": "Core Evolution",
                "items": [
                    {"key": "generations", "label": "Generations", "type": "int", "default": 5, "min": 1, "max": 1000, "step": 1, "help": "Number of evolutionary generations to run."},
                    {"key": "pop_size", "label": "Population Size", "type": "int", "default": 100, "min": 10, "max": 2000, "step": 10, "help": "Number of programs per generation."},
                    {"key": "tournament_k", "label": "Tournament K", "type": "int", "default": 10, "min": 2, "max": 200, "step": 1, "help": "Contestants per tournament when selecting parents."},
                    {"key": "elite_keep", "label": "Elites Kept", "type": "int", "default": 1, "min": 0, "max": 20, "step": 1, "help": "Top programs copied unchanged to next generation."},
                    {"key": "hof_size", "label": "HOF Size", "type": "int", "default": 20, "min": 1, "max": 200, "step": 1, "help": "Max Hall of Fame entries to keep."},
                    {"key": "hof_per_gen", "label": "HOF Per Gen", "type": "int", "default": 3, "min": 0, "max": 50, "step": 1, "help": "Top candidates added to HOF each generation (subject to filters)."},
                    {"key": "seed", "label": "Random Seed", "type": "int", "default": 42, "min": 0, "max": 2**31-1, "step": 1, "help": "Base seed for reproducibility."},
                    {"key": "fresh_rate", "label": "Fresh Rate", "type": "float", "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Fraction of population replaced with brand new programs each generation."},
                    {"key": "p_mut", "label": "Mutation Prob.", "type": "float", "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Probability to mutate a child after selection/crossover."},
                    {"key": "p_cross", "label": "Crossover Prob.", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Probability to crossover parents (0 uses cloning+mutation only)."},
                ],
            },
            {
                "title": "Selection & Ramping",
                "items": [
                    {"key": "selection_metric", "label": "Selection Metric", "type": "select", "default": "ramped", "choices": ["ramped", "fixed", "ic", "auto", "phased"], "help": "Parent selection criterion: ramped fitness, fixed final weights, pure IC, or mixed schedules."},
                    {"key": "ramp_fraction", "label": "Ramp Fraction", "type": "float", "default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Portion of total generations to gradually reach full penalty/weighting."},
                    {"key": "ramp_min_gens", "label": "Ramp Min Gens", "type": "int", "default": 5, "min": 0, "max": 100, "step": 1, "help": "Minimum generations for ramp to avoid premature exploitation."},
                    {"key": "ic_phase_gens", "label": "IC‑only Warmup Gens", "type": "int", "default": 0, "min": 0, "max": 100, "step": 1, "help": "Use pure IC for this many early generations (phased)."},
                    {"key": "rank_softmax_beta_floor", "label": "Rank Softmax Beta (Floor)", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "help": "Initial temperature for rank-based tournament sampling."},
                    {"key": "rank_softmax_beta_target", "label": "Rank Softmax Beta (Target)", "type": "float", "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1, "help": "Final temperature after ramp completes."},
                ],
            },
            {
                "title": "Novelty & Correlation",
                "items": [
                    {"key": "novelty_boost_w", "label": "Novelty Boost", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Bonus weight for low correlation vs Hall of Fame predictions."},
                    {"key": "novelty_struct_w", "label": "Structural Novelty", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Bonus for structural opcode-set dissimilarity vs HOF entries."},
                    {"key": "hof_corr_mode", "label": "HOF Corr Mode", "type": "select", "default": "flat", "choices": ["flat", "per_bar"], "help": "How to compute correlation vs HOF: flat (all points) or per-bar averaged."},
                    {"key": "corr_penalty_w", "label": "Correlation Penalty", "type": "float", "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Penalty weight for correlation vs HOF (applied in fitness)."},
                    {"key": "corr_cutoff", "label": "HOF Corr Cutoff", "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Drop candidates too correlated with HOF beyond this threshold."},
                ],
            },
            {
                "title": "Fitness Weights",
                "items": [
                    {"key": "sharpe_proxy_w", "label": "Sharpe Proxy Weight", "type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Weight for Sharpe-like proxy in fitness (0 uses IC-only)."},
                    {"key": "ic_std_penalty_w", "label": "IC Std Penalty", "type": "float", "default": 0.10, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Penalty weight for IC volatility."},
                    {"key": "turnover_penalty_w", "label": "Turnover Penalty", "type": "float", "default": 0.05, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Penalty for high position turnover (reduces trading)."},
                    {"key": "ic_tstat_w", "label": "IC t‑stat Weight", "type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Include IC t‑stat to reward stability in rank correlations."},
                    {"key": "temporal_decay_half_life", "label": "Temporal Decay Half‑life", "type": "float", "default": 0.0, "min": 0.0, "max": 10000.0, "step": 1.0, "help": "Exponential half-life in bars to weight recent data more (0 disables)."},
                    {"key": "factor_penalty_w", "label": "Factor Neutral Penalty", "type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Weight applied to style-factor exposure penalty."},
                    {"key": "factor_penalty_factors", "label": "Factor List", "type": "text", "default": "ret1d_t,vol20_t,range_rel_t", "help": "Comma-separated factor names to neutralize (matches evaluation features)."},
                    {"key": "stress_penalty_w", "label": "Stress Penalty Weight", "type": "float", "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01, "help": "Penalty weight for stress/transaction-cost robustness."},
                    {"key": "stress_fee_bps", "label": "Stress Fee (bps)", "type": "float", "default": 5.0, "min": 0.0, "max": 100.0, "step": 0.5, "help": "Additional fee applied in stress fitness (per side)."},
                    {"key": "stress_slippage_bps", "label": "Stress Slippage (bps)", "type": "float", "default": 2.0, "min": 0.0, "max": 100.0, "step": 0.5, "help": "Extra slippage applied in stress fitness."},
                    {"key": "stress_shock_scale", "label": "Stress Shock Scale", "type": "float", "default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1, "help": "Multiplier for downside shocks in stress fitness."},
                    {"key": "stress_tail_fee_bps", "label": "Tail Stress Fee (bps)", "type": "float", "default": 10.0, "min": 0.0, "max": 200.0, "step": 0.5, "help": "Heightened fee applied in tail stress scenario."},
                    {"key": "stress_tail_slippage_bps", "label": "Tail Stress Slippage (bps)", "type": "float", "default": 3.5, "min": 0.0, "max": 200.0, "step": 0.5, "help": "Heightened slippage applied in tail stress scenario."},
                    {"key": "stress_tail_shock_scale", "label": "Tail Shock Scale", "type": "float", "default": 2.5, "min": 1.0, "max": 6.0, "step": 0.1, "help": "Amplified downside shock multiplier for tail stress."},
                    {"key": "transaction_cost_bps", "label": "Baseline TC (bps)", "type": "float", "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5, "help": "Baseline transaction-cost assumption applied to turnover."},
                ],
            },
            {
                "title": "Evaluation Windows",
                "items": [
                    {"key": "evaluation_horizons", "label": "Evaluation Horizons", "type": "text", "default": "1", "help": "Comma-separated prediction horizons (in bars) used during evaluation."},
                    {"key": "use_train_val_splits", "label": "Use Train/Val Splits", "type": "bool", "default": True, "help": "Split evaluation window into train/validation segments."},
                    {"key": "train_points", "label": "Train Bars", "type": "int", "default": 840, "min": 0, "max": 100000, "step": 10, "help": "Number of bars allocated to training slice when splits enabled."},
                    {"key": "val_points", "label": "Validation Bars", "type": "int", "default": 360, "min": 0, "max": 100000, "step": 10, "help": "Number of bars allocated to validation slice when splits enabled."},
                    {"key": "regime_diagnostic_factors", "label": "Regime Diagnostic Factors", "type": "text", "default": "regime_volatility_t,regime_momentum_t,cross_btc_momentum_t,sector_momentum_diff_t,onchain_activity_proxy_t,onchain_velocity_proxy_t,onchain_whale_proxy_t", "help": "Comma-separated features tracked in regime diagnostics."},
                ],
            },
            {
                "title": "Multi‑Objective & Fidelity",
                "items": [
                    {"key": "moea_enabled", "label": "Pareto Selection (MOEA)", "type": "bool", "default": False, "help": "Enable multi-objective selection (NSGA‑II‑like) for elites."},
                    {"key": "moea_elite_frac", "label": "Pareto Elite Fraction", "type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Portion of next generation chosen from the first Pareto front."},
                    {"key": "mf_enabled", "label": "Multi‑Fidelity Eval", "type": "bool", "default": False, "help": "Enable cheap first pass on truncated data then re‑evaluate top‑K fully."},
                    {"key": "mf_initial_fraction", "label": "MF Initial Fraction", "type": "float", "default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05, "help": "Fraction of bars used in the cheap first-pass evaluation."},
                    {"key": "mf_promote_fraction", "label": "MF Promote Fraction", "type": "float", "default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05, "help": "Fraction of population promoted to full evaluation."},
                    {"key": "mf_min_promote", "label": "MF Min Promote", "type": "int", "default": 8, "min": 1, "max": 200, "step": 1, "help": "Minimum number promoted regardless of fraction."},
                ],
            },
            {
                "title": "Quality Diversity Archive",
                "items": [
                    {"key": "qd_archive_enabled", "label": "Enable QD Archive", "type": "bool", "default": False, "help": "Maintain a MAP-Elites style archive alongside the Hall of Fame."},
                    {"key": "qd_turnover_bins", "label": "QD Turnover Bins", "type": "text", "default": "0.1,0.3,0.6", "help": "Comma-separated turnover breakpoints for QD descriptors."},
                    {"key": "qd_complexity_bins", "label": "QD Complexity Bins", "type": "text", "default": "0.25,0.5,0.75", "help": "Comma-separated program-size breakpoints for QD descriptors."},
                    {"key": "qd_max_entries", "label": "QD Max Entries", "type": "int", "default": 256, "min": 1, "max": 10000, "step": 1, "help": "Maximum archive cells to keep (older cells evicted)."},
                ],
            },
            {
                "title": "Cross‑Validation",
                "items": [
                    {"key": "cv_k_folds", "label": "CV K Folds", "type": "int", "default": 0, "min": 0, "max": 20, "step": 1, "help": "Use K>1 to enable CPCV‑style purged cross‑validation."},
                    {"key": "cv_embargo", "label": "CV Embargo (bars)", "type": "int", "default": 0, "min": 0, "max": 1000, "step": 1, "help": "Bars to embargo around each validation fold to reduce leakage."},
                ],
            },
            {
                "title": "Backtest Ensemble",
                "items": [
                    {"key": "ensemble_mode", "label": "Ensemble Backtest", "type": "bool", "default": False, "help": "Also backtest an ensemble of top alphas."},
                    {"key": "ensemble_size", "label": "Ensemble Size", "type": "int", "default": 0, "min": 0, "max": 100, "step": 1, "help": "Number of top alphas to include (0 disables)."},
                    {"key": "ensemble_max_corr", "label": "Ensemble Max Corr", "type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Target maximum pairwise correlation when selecting ensemble members."},
                ],
            },
            {
                "title": "Data & Run",
                "items": [
                    {"key": "data_dir", "label": "Data Directory", "type": "text", "default": "./data", "help": "Path to directory containing input CSV files."},
                    {"key": "bt_top", "label": "Backtest Top‑N", "type": "int", "default": 10, "min": 1, "max": 100, "step": 1, "help": "Number of evolved alphas to backtest and summarize."},
                    {"key": "no_clean", "label": "Keep Run Artefacts", "type": "bool", "default": False, "help": "Do not clean previous pipeline runs before starting."},
                    {"key": "dry_run", "label": "Dry Run", "type": "bool", "default": False, "help": "Print resolved configs and planned outputs, then exit."},
                    {"key": "out_summary", "label": "Write Summary JSON", "type": "bool", "default": True, "help": "Write SUMMARY.json with key artefacts for UI consumption."},
                ],
            },
        ],
    })
