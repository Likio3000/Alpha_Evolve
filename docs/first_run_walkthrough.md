# First Run Walkthrough

This mini-tour shows the quickest way to evolve a set of alphas, inspect the
results, and keep the workspace tidy.

> ⚠️ Legacy note: the CLI commands below are kept for archival reference. In
> current builds use the dashboard UI or `/api/pipeline/run` instead.

## 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional quality gates:

```bash
sh scripts/typecheck        # mypy (fast, catch obvious typing mistakes)
sh scripts/test             # pytest -q
```

## 2. Acquire data

Copy your CSVs into `data/` (one symbol per file with columns
`time,open,high,low,close`) **or** fetch a quick SP500 sample:

```bash
uv run scripts/fetch_sp500_data.py --out data_sp500 --years 5
```

## 3. Kick off a pipeline run

```bash
uv run run_pipeline.py 5 \
  --config configs/sp500.toml \
  --data_dir data_sp500 \
  --output-dir ~/.alpha-evolve/runs
```

Key artefacts land inside the chosen `--output-dir` (default:
`pipeline_runs_cs/`). Inspect the latest run via:

```bash
ls ~/.alpha-evolve/runs/run_*/
```

Useful files:

- `SUMMARY.json` – high-level metrics + handy paths for the dashboard
- `backtest_portfolio_csvs/backtest_summary_topN.csv` – per-alpha metrics
- `diagnostics.json` – evolution diagnostics/telemetry
- `meta/` – frozen configs + metadata for reproducibility

## 4. Launch the dashboard

```bash
AE_PIPELINE_DIR=~/.alpha-evolve/runs uv run scripts/run_dashboard.py
# Open http://127.0.0.1:8000/ui
```

The UI can start new runs, stream logs, show diagnostics, and surface the
backtest table and alpha equity curves.

## 5. Housekeeping

Over time, runs add up. Prune older ones or monitor usage with:

```bash
python scripts/cleanup_runs.py --base-dir ~/.alpha-evolve/runs --keep 10 --show-size
# Add --delete to actually remove the listed directories.
```

That’s all you need for a first spin. For deeper dives see
`docs/backend_api.md` and the comments in the config files under `configs/`.
