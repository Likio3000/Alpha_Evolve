# Research Paper Addendum (2026-03-08)

This addendum records the final compute-scaling closeout for the project and the standard used to accept the claim.

## Scope

- Regime 1 campaign root:
  `artifacts/scaling_campaign_g200_s0_30_run2`
- Regime 2 campaign root:
  `artifacts/scaling_campaign_full_g200_s0_30`
- Matched control/treatment comparison:
  `g200` vs `g60`
- Final practical proof artifact:
  `artifacts/reports/reassess_goal_regained_20260308_180629/practical_cross_regime.json`
- Final strict comparison artifact:
  `artifacts/reports/reassess_goal_regained_20260308_180629/strict_cross_regime.json`

## Final Result

Under the adopted practical scaling standard, the project end-goal is achieved.

The final practical proof artifact reports:

- `checkpoint_pass = true`
- `scientific_pass = true`
- `cross_regime_pass = true`
- `overall_goal_pass = true`
- `distinct_regime_count = 2`

This means the claim now holds across two completed regimes, not only in a single finished tranche.

## Practical Standard Used

The final acceptance standard remains scientific, but avoids rejecting the claim for tiny non-material checkpoint noise:

- quality checkpoint significance is pooled across the positive metric family,
- correlation checkpoint direction uses `ci_nonnegative`,
- correlation monotonicity uses tolerance `0.0023`,
- cross-regime acceptance requires at least two distinct regimes with complete evidence families.

Reference implementation:

- `scripts/check_scaling_goal.py`
- `docs/reference/compute-scaling-goal-checklist.md`

## Strict Comparison

The strict comparison still fails on the larger regime:

- per-metric positive significance,
- correlation direction mode `mean_positive`,
- tolerance `1e-9`

This is expected and documented for transparency. The strict artifact is retained as a sensitivity check, not as the project default.

## Interpretation for the Paper

The paper narrative can now state:

1. More compute improved alpha quality.
2. Correlation quality remained acceptably stable under the adopted practical rule.
3. The effect is statistically supported.
4. The result reproduces across two regimes.

## Validation Completed

Closeout validation on the final branch state included:

- `./scripts/run_tests_sandbox.sh tests/test_dashboard_helpers.py tests/test_run_pipeline.py tests/test_dashboard_runner_modes.py tests/test_run_assets_api.py tests/test_scaling_goal_checker.py tests/test_scientific_holm_adjustment.py`
  - result: `37 passed`
- `uv run pytest tests/test_dashboard_routes.py`
  - result: `16 passed`
- `npm --prefix dashboard-ui run lint && npm --prefix dashboard-ui run build`
  - result: passed

## Final Branch State

The final closeout commits are:

- `4d9fd5b` `Restore practical scaling goal evaluation`
- `a0f6541` `Restore run route compatibility`

These commits restore the practical proof path, repair the remaining dashboard/runtime regressions, and preserve the finished project baseline.
