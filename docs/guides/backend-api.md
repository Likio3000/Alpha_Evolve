# Dashboard API Guide

The dashboard server exposes a JSON API for launching pipeline jobs, browsing run artefacts, and powering custom dashboards. All routes are served from the Django app launched via `uv run scripts/run_dashboard.py`.

Base URL defaults to `http://127.0.0.1:8000`.

## Health
- `GET /health` → `{ "ok": true }` when the server is alive.

## Pipeline Control
- `POST /api/pipeline/run`
  - Body: `PipelineRunRequest` JSON
    ```json
    {
      "generations": 5,
      "dataset": "sp500",
      "config": "optional path to TOML",
      "data_dir": "optional/path/to/csvs",
      "overrides": {
        "bt_top": 10,
        "disable_align_cache": true
      }
    }
    ```
  - Response: `{ "job_id": "uuid" }`
  - Behaviour: spawns `run_pipeline.py` in a background process, streams log/progress messages into an in-memory queue, persists UI context under `<run>/meta/ui_context.json` once the job completes.

- `GET /api/job-status/<job_id>` → `{ "exists": bool, "running": bool }`
- `GET /api/job-log/<job_id>` → `{ "log": "…\n…" }` (monotonic log buffer)

Use these endpoints to poll progress when not using the bundled UI.

## Run Catalogue
- `GET /api/runs?limit=50`
  - Returns a list of recent runs with `{ "path", "name", "label", "sharpe_best" }`.
  - `path` is relative to the repo when possible so the UI can deep-link.

- `GET /api/last-run`
  - Returns `{ "run_dir": "path", "sharpe_best": float | null }` using the pointer stored in `pipeline_runs_cs/LATEST`.

- `POST /api/run-label`
  - Body: `{ "path": "pipeline_runs_cs/run_...", "label": "Optional note" }`
  - Persists labels in `.run_labels.json` under the pipeline directory.

## Artefact Access
- `GET /api/backtest-summary?run_dir=run_...`
  - Outputs an array mirroring `backtest_summary_topN.csv` rows (`AlphaID`, `Sharpe`, drawdown stats, etc.).
- `GET /api/alpha-timeseries?run_dir=run_...&alpha_id=Alpha_01`
  - Returns `{ "date": [...], "equity": [...], "ret_net": [...] }` extracted from the per-alpha CSV.
- `GET /api/run-asset?run_dir=run_...&file=backtest_portfolio_csvs/backtest_summary_top1.csv`
  - Streams any file inside the run directory (PNG downloads, CSVs); optional `sub` parameter helps when the UI stores the asset folder separately.

## Configuration Metadata
The UI builds forms from the following helper endpoints:
- `GET /ui-meta/evolution-params` – documentation for `EvolutionParams` knobs.
- `GET /ui-meta/pipeline-params` – categorised descriptions for pipeline/backtest options.
- `GET /api/config/defaults` – JSON dump of `EvolutionConfig` and `BacktestConfig` defaults (flattened for the UI).
- `GET /api/config/presets` – list of available TOML presets under `configs/`.
- `GET /api/config/preset-values?name=sp500` – resolved values for a preset.
- `POST /api/config/save` – persist a config snapshot (writes under `configs/generated/` by default).

## Event Streams
The server maintains an internal SSE queue per job. The helper endpoint `run_pipeline.sse_events(job_id)` is now exposed at `GET /api/pipeline/events/<job_id>` and streams `data: { ... }` payloads with `log`, `progress`, `diag`, and `score` events. You can still compose your own response via `scripts/dashboard_server/helpers.make_sse_response` if you need custom dispatching logic.

To stop a running pipeline from the UI or scripts, send `POST /api/pipeline/stop/<job_id>`.

## Versioning Notes
- Payloads accept extra keys; they are forwarded as CLI flags when possible.
- Every artefact path is validated to stay under `AE_PIPELINE_DIR` to avoid path traversal.
- When you introduce new files under a run, update both this guide and the UI metadata endpoints to keep integrations aligned.
