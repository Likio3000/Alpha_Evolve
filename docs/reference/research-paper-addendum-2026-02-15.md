# Research Paper Addendum (2026-02-15)

This addendum updates the paper narrative with the latest matched-seed scientific runs completed on February 15, 2026.

## Scope

- Dataset/pipeline regime: `data_sp500_small`, quick benchmark harness.
- Matched seeds: `58..65` (8 paired seeds).
- Main compute scaling test:
  - Control: `g60` run roots in `artifacts/science_scale_fix_v8_g60ck_s58_65/merged_runs`
  - Treatment: `g200` run roots in `artifacts/science_scale_fix_v8_g200ck_s58_65/merged_runs`
  - Scientific compare JSON:
    `artifacts/science_scale_fix_v8_analysis/scientific_g200_vs_g60_s58_65.json`

## Main Result: More Compute Improved Ensemble Quality

For `g200 vs g60`, we observed statistically significant improvements on the ensemble-level metrics:

- `ensemble_sharpe`: mean improvement `+0.0932`, CI95 `[+0.0448, +0.1409]`, one-sided `p=0.0156`, pass.
- `ensemble_annret`: mean improvement `+0.0147`, CI95 `[+0.00693, +0.02249]`, one-sided `p=0.0156`, pass.
- `ensemble_maxdd` (improvement means lower drawdown): mean improvement `+0.00619`, CI95 `[+0.00224, +0.01188]`, one-sided `p=0.0156`, pass.
- `pair_mean_abs_corr` (improvement means lower pairwise correlation): mean improvement `+0.00676`, CI95 `[+0.000164, +0.01330]`, one-sided `p=0.04297`, pass.

Non-significant in this tranche:

- `best_sharpe`: positive mean delta (`+0.0260`) but not significant (`p=0.125`).

## Checkpoint Dynamics in the 200-Generation Runs

From `artifacts/science_scale_fix_v8_analysis/g200ck_s58_65_checkpoint_summary.json`:

- Strong/significant checkpoint step at `20 -> 40` for ensemble Sharpe and selected-correlation metrics.
- Later checkpoint increments show diminishing returns in this 8-seed sample, with mostly non-significant increments.

Interpretation:

- The long-run advantage is clear at the final `g200 vs g60` comparison.
- Incremental per-checkpoint gains flatten after the early/mid run stages for this sample size.

## Plateau-Strategy A/B Outcome (v4 vs v3 at g60)

We tested the new compute-invariant plateau strategy (`v4`) against `v3` at fixed `g60`:

- Compare JSON:
  `artifacts/science_scale_fix_v9_analysis/scientific_g60_v4_vs_v3_s58_65.json`
- Report:
  `artifacts/science_scale_fix_v9_analysis/v4_g60_ab_report_s58_65.md`

Result:

- No primary metric passed significance in favor of `v4` on this tranche.
- Directionally, `v4` underperformed `v3` on ensemble Sharpe and annual return in this sample.

Paper implication:

- Keep the compute-scaling claim (`g200 > g60`) for ensemble quality.
- Do not claim plateau-v4 superiority yet; treat it as exploratory and pending larger-sample retuning.

## Reproducibility Pointers

- `g200 vs g60` report:
  `artifacts/science_scale_fix_v8_analysis/g200_vs_g60_report_s58_65.md`
- `g200` checkpoint summary:
  `artifacts/science_scale_fix_v8_analysis/g200ck_s58_65_checkpoint_summary.json`
- `g60` checkpoint summary:
  `artifacts/science_scale_fix_v8_analysis/g60ck_s58_65_checkpoint_summary.json`
- `v4 vs v3` report:
  `artifacts/science_scale_fix_v9_analysis/v4_g60_ab_report_s58_65.md`
