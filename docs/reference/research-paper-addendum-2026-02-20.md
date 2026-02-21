# Research Paper Addendum (2026-02-20)

This addendum records the protocol upgrade used for the next large-sample compute-scaling tranche.

## Why this update

The project goal is now evaluated with explicit scientific gates, not only point estimates:

- compute increase should improve quality metrics,
- correlation quality should not deteriorate,
- significance should pass one-sided tests with Holm adjustment where applicable,
- results should reproduce across multiple regimes/windows.

## Protocol changes since 2026-02-15

1. Parallel multi-fidelity evaluation now respects configured workers (cheap + promoted full pass in pool mode).
2. Scientific scripts emit Holm-adjusted one-sided p-values in checkpoint/scientific reports.
3. A formal goal-gate checker was added:
   - `scripts/check_scaling_goal.py`
   - acceptance checklist: `docs/reference/compute-scaling-goal-checklist.md`

## Ongoing large-sample tranche

Primary campaign (in progress at authoring time):

- config: `configs/bench_sp500_scaling_monotonic_v4.toml`
- seeds: `0:30` (30 seeds)
- generations/checkpoints: `200` with `30,60,90,120,200`
- output root: `artifacts/scaling_campaign_g200_s0_30_run2`

## Paper-facing requirement

After this tranche finishes and gates are evaluated:

1. publish artifact paths + gate output JSON in a follow-up addendum,
2. update the main paper narrative (`Alpha_evolve_paper.pdf` source materials) to reflect:
   - compute-scaling evidence,
   - correlation stability evidence,
   - statistical method details (Holm-adjusted one-sided tests),
   - robustness across regimes.
