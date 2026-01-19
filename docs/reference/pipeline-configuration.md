# Pipeline Configuration Reference

Pipeline behaviour is controlled by two dataclasses exposed via `alpha_evolve.config`:
- `EvolutionConfig` – evolutionary search parameters.
- `BacktestConfig` – portfolio simulation and evaluation settings.

Additional heuristics live in `alpha_evolve.programs.utils.EvolutionParams` and are surfaced to the UI through `/ui-meta/evolution-params`.

Use this reference as the canonical description of the knobs exposed through the dashboard, REST API, and TOML presets.

## Evolution Basics
- `generations` – number of generations to evolve.
- `pop_size` – candidates per generation.
- `seed` – RNG seed for reproducibility.
- `fresh_rate` – fraction of population replaced by brand new programs each iteration.
- `p_mut`, `p_cross` – probabilities for mutation and crossover.
- `elite_keep` – count of elites copied straight into the next generation.
- `hof_size`, `hof_per_gen` – hall-of-fame capacity and per-generation additions.

## Selection & Ramping
- `selection_metric` – ranking strategy (`ramped`, `fixed`, `ic`, `lcb`, `psr`, `sharpe`, `auto`, `phased`).
- `ramp_fraction`, `ramp_min_gens` – progression speed from exploratory to final weights.
- `ic_phase_gens` – IC-only warmup for the phased schedule.
- `rank_softmax_beta_floor/target` – temperature bounds for rank-based sampling.

## Novelty & Correlation Control
- `novelty_boost_w`, `novelty_struct_w` – reward novel predictions and operator structures versus the hall of fame.
- `corr_penalty_w`, `corr_cutoff` – penalise or filter near-duplicates.
- `hof_corr_mode` – correlation computed on flattened or per-bar timeseries.

## Fitness Weights
- `sharpe_proxy_w` – weight for the Sharpe-like proxy during evolution.
- `ic_std_penalty_w` – penalise volatile IC.
- `ic_tstat_w` – emphasise IC stability.
- `turnover_penalty_w` – discourage high turnover.
- `temporal_decay_half_life` – exponential decay applied to recent data.
- `factor_penalty_w` + `factor_penalty_factors` – penalise exposure to named style factors.
- `stress_penalty_w`, `stress_fee_bps`, `stress_slippage_bps`, `stress_shock_scale` – stress test aggressiveness.

## Evaluation Windows & Fidelity
- `evaluation_horizons` – list of horizons (bars) to evaluate per generation.
- `use_train_val_splits` – enable separate validation slices.
- `train_points`, `val_points` – sample counts when splits are active.
- `mf_enabled`, `mf_initial_fraction`, `mf_promote_fraction`, `mf_min_promote` – multi-fidelity evaluation controls.
- `moea_enabled`, `moea_elite_frac` – toggle Pareto-based selection.

## Quality Diversity Archive
- `qd_archive_enabled` – enable the MAP-Elites archive.
- `qd_turnover_bins`, `qd_complexity_bins` – bin edges for descriptors.
- `qd_max_entries` – max archive size.

## Cross-Validation
- `cv_k_folds` – >1 enables CPCV-like purged cross-validation.
- `cv_embargo` – embargo window (bars) around each fold.

## Backtest Options
- `bt_top` – number of top alphas to backtest.
- `long_short_n` – restrict to the top/bottom N ranked symbols per rebalance (0 uses all).
- `sector_neutralize_positions` – retain sector/industry neutrality.
- `fees_bps`, `slippage_bps` – baseline trading costs.
- `rebalance_every` – rebalance cadence in bars.
- `max_position_change` – optional turnover clamp per step.
- `ensemble_mode`, `ensemble_size`, `ensemble_max_corr` – ensemble backtesting parameters.

## Data Management
- `data_dir` – directory containing input CSVs.
- `max_lookback_data_option` – controls alignment horizon.
- `disable_align_cache`, `align_cache_dir` – toggle alignment caching.
- `no_clean` – retain previous pipeline outputs.
- `out_summary` – persist `SUMMARY.json` when running programmatically.

## EvolutionParams (Generation Bias)
Located in `src/alpha_evolve/programs/utils.py`:
- `vector_ops_bias` – probability of forcing vector outputs when sampling ops.
- `relation_ops_weight`, `cs_ops_weight`, `default_op_weight` – relative weights for relation and cross-sectional ops.
- `max_setup_ops`, `max_predict_ops`, `max_update_ops` – soft limits mirrored by config defaults.
- `ops_split_base` – base distribution of operations across (setup, predict, update) blocks.
- `ops_split_jitter` – randomness injected when seeding fresh programs.

These heuristics bias the operator sampler without changing the core search space. The dashboard exposes friendly controls based on `/ui-meta/evolution-params`.

## Editing & Presets
- TOML presets under `configs/` hydrate directly into `EvolutionConfig`/`BacktestConfig`.
- Environment variables can override individual fields when launching the dashboard or CLI (precedence: config file < env vars < CLI flags/JSON values).
- Any extra keys submitted to `/api/pipeline/run` go through `build_pipeline_args` and become `--flag value` pairs when compatible with the CLI parser. Non-scalar entries are ignored.

When you introduce new configuration options:
1. Add them to the relevant dataclass in `src/alpha_evolve/config/model.py` (re-exported via `alpha_evolve.config`).
2. Update this reference and the metadata emitters in `scripts/dashboard_server/ui_meta.py`.
3. Extend tests (e.g. `tests/test_dashboard_routes.py`) to cover default serialization.
