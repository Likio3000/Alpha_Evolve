# Compute-Scaling Goal Checklist

This checklist defines when we can claim the project end-goal is achieved:

- More compute yields better alpha quality.
- Correlation quality stays stable or improves (no deterioration).
- The effect is statistically supported (one-sided tests, Holm-adjusted when available).

## Acceptance Gates

Use these gates on each regime tranche:

1. Pairwise checkpoint gates (within long runs, e.g. `30 -> 60 -> 90 -> 120 -> 200`)
- Required quality metrics: `ensemble_portfolio_sharpe`, `best_backtest_sharpe`.
- Required correlation metric: `selected_avg_abs_corr` (lower is better).
- Relaxed default (diminishing-returns aware):
  - quality metrics:
    - at least `25%` of adjacent steps are statistically significant,
    - `100%` of adjacent steps keep the correct direction (`mean_improvement > 0`),
    - final adjacent step keeps correct direction.
  - correlation metrics:
    - significance fraction requirement is `0%` by default,
    - `100%` of adjacent steps keep correct direction (`mean_improvement > 0`),
    - final adjacent step keeps correct direction.
- The checker remains configurable if we want stricter or looser thresholds.

2. Monotonic + trend gates on generation-level means
- Quality metrics must be non-decreasing across checkpoints.
- Correlation metric must be non-increasing across checkpoints.
- Oriented trend slope must be positive with one-sided permutation `p <= 0.05`.

3. Matched control/treatment scientific gates (optional but recommended)
- Typical comparison: `g200` vs `g60` on matched seeds.
- Required metrics: `ensemble_sharpe`, `ensemble_annret`, `pair_mean_abs_corr`.
- Same directional + CI + p-value gate as above.

4. Cross-regime robustness
- Run the same protocol on at least two regimes/windows.
- Claim is accepted only if all required gates pass in each regime tranche.

## Repro Commands

### 1) Launch a long checkpoint campaign

```bash
bash scripts/benchmark_sp500_parallel.sh \
  --mode full \
  --config configs/bench_sp500_scaling_monotonic_v4.toml \
  --seeds 0:30 \
  --jobs 6 \
  --outdir artifacts/scaling_campaign_g200_s0_30 \
  -- --generations 200 --checkpoint-gens 30,60,90,120,200 --skip-plots
```

Or run multiple regimes with one command:

```bash
bash scripts/run_scaling_regime_campaign.sh \
  --regime small:configs/bench_sp500_scaling_monotonic_v4.toml \
  --regime full:configs/bench_sp500_scaling_monotonic_v4_full.toml \
  --seeds 0:30 --jobs 6 --generations 200 --checkpoints 30,60,90,120,200 \
  --outdir artifacts/scaling_regimes
```

### 2) Aggregate shards

```bash
uv run python scripts/aggregate_parallel_benchmarks.py \
  --root artifacts/scaling_campaign_g200_s0_30
```

### 3) Optional matched-group compare (example: g200 vs g60)

```bash
uv run python scripts/scientific_compare.py \
  --control-root <g60_merged_runs_root> \
  --treatment-root <g200_merged_runs_root> \
  --out artifacts/reports/scientific_g200_vs_g60.json
```

### 4) Evaluate goal gates

```bash
uv run python scripts/check_scaling_goal.py \
  --checkpoint-summary-json artifacts/scaling_campaign_g200_s0_30/aggregate/checkpoint_summary_combined.json \
  --scientific-json artifacts/reports/scientific_g200_vs_g60.json \
  --out artifacts/reports/scaling_goal_check_g200_s0_30.json
```

Exit code `0` means all gates passed. Exit code `2` means one or more gates failed.

## Paper Update Requirement

When a tranche passes all gates:

- Update the next research addendum in `docs/reference/`.
- Update the main narrative paper (`Alpha_evolve_paper.pdf` source materials) to include:
  - compute-vs-quality results,
  - correlation stability results,
  - Holm-adjusted significance details,
  - regime robustness results.
