# Run Artefacts

Each pipeline execution creates `pipeline_runs_cs/run_<timestamp>/` (or inside the directory specified by `AE_PIPELINE_DIR`). The dashboard and scripts rely on the following structure.

## Top-Level Files
- `SUMMARY.json` – machine-readable snapshot with `schema_version`, top Sharpe metrics, and relative paths to key files.
- `diagnostics.json` – per-generation metrics and cache utilisation records (`PROGRESS`/`DIAG` lines rendered by the dashboard).
- `pipeline.log` (optional) – combined stdout/stderr when logging to file is enabled.

## Backtest Outputs (`backtest_portfolio_csvs/`)
- `backtest_summary_topN.csv` / `.json` – aggregated metrics for the top-N alphas (Sharpe, returns, drawdown, turnover, operation count, program string).
- `alpha_XX_timeseries.csv` – per-alpha equity curve with `date`, `equity`, and `ret_net` columns.
- `alpha_XX_timeseries.png` – optional chart generated when plotting dependencies are installed.

## Meta Information (`meta/`)
- `evolution_config.json` – resolved `EvolutionConfig` used for the run.
- `backtest_config.json` – resolved `BacktestConfig` for the run.
- `run_metadata.json` – provenance (seed, git SHA, data alignment summary, start/end times).
- `hof_gen_*.json` – hall-of-fame snapshots per generation (present when `persist_hof_per_gen` is enabled).
- `ui_context.json` – JSON payload captured by the dashboard when a run starts (job metadata, submitted overrides).
- `data_alignment.json` – loader diagnostics (symbols kept/discarded, overlap windows) when available.

## Diagnostics & Aux Files
- `diagnostics/` – optional rendered plots and cached summaries.
- `programs/` – serialized alpha representations for deeper inspection.
- `cache/` – alignment cache or intermediate arrays when enabled.

## Maintenance Scripts
- `scripts/cleanup_runs.py` trims these directories while respecting the latest pointer (`pipeline_runs_cs/LATEST`).
- `scripts/monitor_logs.sh` tails run logs when jobs are long-lived.

Always keep artefacts self-contained: when adding new files, update this reference and ensure the dashboard exposes or ignores them intentionally.
