# Alpha Evolve Orientation

This guide gives new contributors (including AI coding agents) the context they need to land changes responsibly.

## Mission
- **Purpose:** Continuously evolve quantitative trading signals (alphas) that survive backtesting and diagnostic scrutiny.
- **Success signals:** Sharpe ratio stability, controlled drawdowns, turnover discipline, and diversified correlations across runs.
- **Core assets:** The evolution pipeline, reproducible run artefacts, and the dashboard that orchestrates both.

## Architecture Snapshot
- **Evolution engine (`python -m alpha_evolve.cli.pipeline` + `src/alpha_evolve/evolution/`):** Drives program generation, evaluation, and selection. Configured by `EvolutionConfig`/`BacktestConfig` in `alpha_evolve.config` and the TOML presets in `configs/`.
- **Backtesting stack (`src/alpha_evolve/backtesting/`, `src/alpha_evolve/utils/`):** Aligns OHLC data, applies transaction-cost and stress adjustments, and writes summaries under each run directory.
- **Dashboard server (`src/alpha_evolve/dashboard/api/`):** Django ASGI app launched via `uv run scripts/run_dashboard.py`. It exposes REST endpoints under `/api/*`, keeps run/job state, and serves the built UI bundle from `dashboard-ui/dist/`.
- **Artefact layout (`pipeline_runs_cs/`):** Each run creates `run_*` folders with configs, diagnostics, and backtest CSVs; the dashboard stores relative pointers and labels here.
- **Supporting scripts (`scripts/`):** Helpers for fetching data, pruning runs, plotting diagnostics, executing smoke tests, and maintaining dependencies.

## Key Directories
- `src/alpha_evolve/programs/` – shared AST and operator utilities used during evolution.
- `src/alpha_evolve/evolution/` – selection, fitness, mutation, and orchestration code.
- `src/alpha_evolve/backtesting/` – portfolio simulation, metrics, and stress testing.
- `scripts/` – operational scripts (dashboard launcher, data fetch, cleanup, smoke runs, tests).
- `configs/` – curated TOML configs (SP500 defaults plus workspace overrides) that mirror the `alpha_evolve.config` dataclasses.
- `dashboard-ui/` – pre-built static SPA; rebuild instructions live in `docs/guides/dashboard-ui.md`.
- `tests/` – pytest coverage for loaders, configs, and dashboard routes.

## Tooling Expectations
- **Python 3.12 + uv:** `uv` is the default runner (`uv run`, `uv pip`). Plain `python` works if `uv` is unavailable.
- **Virtualenv:** Either `scripts/setup_env.sh` or manual `pip install -r requirements.txt`.
- **Data:** Place CSV inputs under `data/` (one symbol per file). `scripts/fetch_sp500_data.py` can seed a sample dataset.
- **Environment variables:** `AE_PIPELINE_DIR` points the dashboard and helpers at an alternate runs directory. Logging for the dashboard obeys `LOG_LEVEL`, `LOG_FILE`, `HOST`, and `PORT`.

## Working Style
- Start by running the dashboard smoke test (`sh scripts/smoke_run.sh`) or the targeted pytest modules you touch.
- Prefer configuration changes through TOML presets so the UI and API remain authoritative; document new knobs.
- Keep artefacts self-describing: if you add files under `pipeline_runs_cs/run_*`, persist metadata in `meta/` and update `docs/reference/run-artifacts.md`.
- When touching `scripts/dashboard_server/`, mirror behaviour in `tests/test_dashboard_routes.py` and verify endpoints stay backward compatible.
- Preserve reproducibility: avoid introducing hidden state, favour dataclass-driven configs, and surface overrides through JSON-friendly types.

## Quick Checklist
1. Can you explain how your change affects the evolution pipeline, backtesting, and the dashboard?
2. Have you updated or added tests plus the relevant documentation page?
3. Do artefacts produced by a new run still contain the files the UI expects (`SUMMARY.json`, `diagnostics.json`, backtest CSVs)?
4. Did you verify logarithmic output via the dashboard or `scripts/run_dashboard.py` logs if background jobs changed?

Use this guide as your compass: each contribution should strengthen the alpha-evolution loop while keeping the surrounding tooling and documentation in sync.
