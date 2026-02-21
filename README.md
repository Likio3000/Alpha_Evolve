# Alpha Evolve

[![Tests](https://github.com/Likio3000/Alpha_Evolve/actions/workflows/python.yml/badge.svg)](https://github.com/Likio3000/Alpha_Evolve/actions/workflows/python.yml)

Alpha Evolve is a research project for evolving cross-sectional trading alphas, backtesting them, and testing whether more compute can produce better and less-correlated alpha sets.

## What This Project Contains

- An evolutionary search engine for alpha programs (`src/alpha_evolve/evolution`).
- A cross-sectional backtester with risk controls and optional ensemble evaluation (`src/alpha_evolve/backtesting`).
- A dashboard API + bundled UI for launching runs and monitoring results (`scripts/run_dashboard.py`, `src/alpha_evolve/dashboard`).
- Reproducibility and analysis tooling for benchmark/scientific runs (`scripts/`, `artifacts/`, `docs/reference/research-paper-addendum-2026-02-15.md`, `docs/reference/compute-scaling-goal-checklist.md`).

## Current Research Status (Documented in Repo)

- Latest documented research updates:
  - `docs/reference/research-paper-addendum-2026-02-15.md`
  - `docs/reference/research-paper-addendum-2026-02-20.md`
- Formal end-goal acceptance gates: `docs/reference/compute-scaling-goal-checklist.md`.
- Latest large-sample tranche (completed February 21, 2026):
  - campaign root: `artifacts/scaling_campaign_g200_s0_30_run2`
  - matched `g200 vs g60` scientific compare (30 seeds):
    `artifacts/scaling_campaign_g200_s0_30_run2/analysis_g200_vs_g60/scientific_g200_vs_g60.json`
  - full goal check (diminishing-returns-aware defaults): pass
    `artifacts/scaling_campaign_g200_s0_30_run2/analysis_g200_vs_g60/scaling_goal_check_full_relaxed_v2.json`
- That tranche shows significant improvement on `ensemble_sharpe`, `ensemble_annret`, and `pair_mean_abs_corr`, with positive `best_sharpe` improvement as well.
- The same addendum reports that plateau strategy `v4` did not beat `v3` in that fixed-sample A/B.
- The main paper PDF (`Alpha_evolve_paper.pdf`) still needs a narrative update to incorporate newer addendum findings.

## Requirements

- Python `3.12+`
- Project dependencies from `requirements.txt`
- Optional: Node.js (only needed if you want to rebuild `dashboard-ui/dist/`)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

or:

```bash
sh scripts/setup_env.sh
```

`uv` is recommended for running commands quickly. If you do not use `uv`, replace `uv run ...` with your Python environment equivalent.

## Quick Start

### 1. Prepare data

Input files are one CSV per symbol with columns:

- `time`
- `open`
- `high`
- `low`
- `close`

`time` can be Unix epoch seconds or ISO8601 timestamps.

Example optional datasets:

```bash
# Full SP500-style example dataset
uv run python scripts/fetch_sp500_data.py --out data_sp500 --years 20

# Small subset for faster local iteration
uv run python scripts/make_sp500_subset.py --out data_sp500_small --tickers 30 \
  --start-date 2020-01-01 --max-rows 756 --min-rows 504
```

### 2. Start dashboard

```bash
AE_PIPELINE_DIR=~/alpha-evolve-runs uv run scripts/run_dashboard.py
```

Then open [http://127.0.0.1:8000/ui/](http://127.0.0.1:8000/ui/).

### 3. Run a pipeline job

Preferred day-to-day flow: run from the dashboard UI.

Programmatic/CLI module usage:

```bash
uv run python -m alpha_evolve.cli.pipeline 10 --config configs/sp500.toml
```

Small/faster config:

```bash
uv run python -m alpha_evolve.cli.pipeline 6 --config configs/sp500_small.toml
```

## Usage Notes

- Console entrypoints are not installed via `[project.scripts]`; run the module directly (`python -m alpha_evolve.cli.pipeline`) or use the dashboard.
- Config precedence is:
  `config file < environment variables < CLI flags`.
- Output run directory defaults to `pipeline_runs_cs/`, overridable by:
  - `--output-dir <path>`
  - `AE_PIPELINE_DIR=<path>` (or `AE_OUTPUT_DIR=<path>`)
- The latest run pointer is written to `pipeline_runs_cs/LATEST` (or equivalent under your configured output root).

## Dashboard + API

Start server:

```bash
uv run scripts/run_dashboard.py
```

Common endpoints:

- `POST /api/pipeline/run`
- `GET /api/job-status/<job_id>`
- `GET /api/job-log/<job_id>`
- `GET /api/runs`
- `GET /api/backtest-summary?run_dir=...`
- `GET /api/alpha-timeseries?run_dir=...&alpha_id=...`
- `GET /api/config/presets`

## Testing

### Sandbox-friendly test loop

Use:

```bash
./scripts/run_tests_sandbox.sh
```

This sets `SKIP_MP_TESTS=1` by default to skip multiprocessing-heavy tests that can hang in constrained sandboxes.

Examples:

```bash
./scripts/run_tests_sandbox.sh -k alpha_timeseries
PYTEST_TIMEOUT=30 ./scripts/run_tests_sandbox.sh tests/test_evaluation_logic.py
```

### Full validation (outside constrained sandbox)

```bash
uv run pytest
```

When multiprocessing behavior changes, include:

```bash
uv run pytest tests/test_dashboard_routes.py
```

## Scientific / Benchmark Workflow

Useful scripts:

- `scripts/benchmark_sp500.py`
- `scripts/benchmark_sp500_parallel.sh`
- `scripts/run_scaling_regime_campaign.sh`
- `scripts/scientific_compare.py`
- `scripts/check_scaling_goal.py`
- `scripts/fit_scaling_laws.py`
- `scripts/generate_scaling_report.py`

See `artifacts/` for generated run bundles and analysis outputs.

Typical long-run evidence flow:

```bash
# 1) Run campaign
bash scripts/benchmark_sp500_parallel.sh \
  --mode full \
  --config configs/bench_sp500_scaling_monotonic_v4.toml \
  --seeds 0:30 \
  --jobs 6 \
  --outdir artifacts/scaling_campaign_g200_s0_30 \
  -- --generations 200 --checkpoint-gens 30,60,90,120,200 --skip-plots

# 2) Aggregate
uv run python scripts/aggregate_parallel_benchmarks.py \
  --root artifacts/scaling_campaign_g200_s0_30

# 3) Compare final compute levels (example g200 vs g60 roots)
uv run python scripts/scientific_compare.py \
  --control-root <g60_runs_root> \
  --treatment-root <g200_runs_root> \
  --out artifacts/reports/scientific_g200_vs_g60.json

# 4) Evaluate goal gates
uv run python scripts/check_scaling_goal.py \
  --checkpoint-summary-json artifacts/scaling_campaign_g200_s0_30/aggregate/checkpoint_summary_combined.json \
  --scientific-json artifacts/reports/scientific_g200_vs_g60.json \
  --out artifacts/reports/scaling_goal_check.json
```

## Repository Pointers

- Main package: `src/alpha_evolve/`
- Config presets: `configs/`
- Dashboard UI source: `dashboard-ui/src/`
- Bundled dashboard build served by backend: `dashboard-ui/dist/`
- Docs index: `docs/README.md`

## License

[MIT](LICENSE)
