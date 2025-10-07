# Getting Started

Use this walkthrough to run Alpha Evolve end-to-end: install dependencies, launch the dashboard, execute a run, and explore the outputs.

## 1. Prerequisites
- Python 3.12+
- `uv` (recommended) or plain `python`
- Target dataset in CSV form (time, open, high, low, close per symbol)

Create and activate a virtual environment, then install packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or rely on the helper script:

```bash
sh scripts/setup_env.sh
```

## 2. Seed Example Data (optional)
Fetch adjusted SP500 data as a quick testbed:

```bash
uv run scripts/fetch_sp500_data.py --out data_sp500 --years 5
```

Point future runs at the generated directory with `--data_dir data_sp500` or by setting it in a config preset.

## 3. Launch the Dashboard Server
Set the pipeline output directory (defaults to `pipeline_runs_cs/` relative to the repo) and start the ASGI app:

```bash
AE_PIPELINE_DIR=~/alpha-evolve-runs \
  uv run scripts/run_dashboard.py
```

Environment knobs:
- `HOST` / `PORT` – network binding (defaults: `127.0.0.1:8000`)
- `LOG_LEVEL` – `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `LOG_FILE` – path to tee logs to disk
- `ACCESS_LOG` – enable Uvicorn request logging when set to `1`

Open http://127.0.0.1:8000/ui to reach the bundled dashboard.

## 4. Start an Evolution Run
From the UI use the “New Run” form. For API or scripting, POST to `/api/pipeline/run`:

```bash
curl -X POST http://127.0.0.1:8000/api/pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{
        "generations": 3,
        "dataset": "sp500",
        "data_dir": "data_sp500",
        "overrides": {
          "bt_top": 5,
          "disable_align_cache": true
        }
      }'
```

The response contains a `job_id`. Use `/api/job-status/<job_id>` or `/api/job-log/<job_id>` to poll progress; the UI streams the same feed automatically.

## 5. Inspect Run Outputs
Each run writes a directory under `AE_PIPELINE_DIR` (e.g. `run_2025-02-01T10-00-00`). Key artefacts:
- `SUMMARY.json` – snapshot of best Sharpe and important file paths.
- `diagnostics.json` – per-generation stats consumed by the dashboard charts.
- `backtest_portfolio_csvs/` – summary CSVs plus per-alpha timeseries.
- `meta/` – frozen configs (`evolution_config.json`, `backtest_config.json`, `run_metadata.json`) and UI payloads (`ui_context.json`).

Use `/api/backtest-summary?run_dir=...` to fetch the top-N alpha metrics and `/api/alpha-timeseries` for equity curves.

## 6. Housekeeping
Runs accumulate quickly. Clean or review them with:

```bash
python scripts/cleanup_runs.py --base-dir ~/alpha-evolve-runs --keep 10 --show-size
# Append --delete to actually remove the oldest runs.
```

Add the repo root to your shell history: `export AE_PIPELINE_DIR=~/alpha-evolve-runs` so future sessions pick up the same location.

## Next Steps
- Review `docs/reference/pipeline-configuration.md` to understand available parameters.
- Explore `docs/reference/run-artifacts.md` for a deeper look at the files under each run.
- Keep the dashboard running while iterating; the API is designed for rapid reruns with small overrides.
