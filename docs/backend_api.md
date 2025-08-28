Backend integration guide (UI ↔ pipeline)

Goals
- Stable, simple contract for the UI to run the pipeline and read results.
- Avoid heavy Python deps on the UI side; consume JSON/CSV artefacts.

Key artefacts
- Runs directory: `pipeline_runs_cs/run_*`
- Manifest: `SUMMARY.json` (schema_version=1, includes key file paths and best metrics)
- Backtest table: `backtest_portfolio_csvs/backtest_summary_topN.{csv,json}`
- Per‑alpha series: `backtest_portfolio_csvs/alpha_XX_timeseries.csv|png`

Programmatic helpers (Python)
- `utils/run_index.py`
  - `list_runs(base_dir=./pipeline_runs_cs) -> List[dict]`
  - `get_latest_run(base_dir=./pipeline_runs_cs) -> Optional[str]`
  - `read_summary(run_dir) -> Optional[dict]`
  - `load_backtest_table(run_dir) -> List[dict]`

CLI entry points (uv/pip)
- Evolve + backtest (all‑in‑one):
  - `uv run run_pipeline.py <generations> [flags...]`
  - `--config <toml>` and `--data_dir <dir>` are the most important flags.

Recommended config for crypto 4h (fast preset)
- `configs/crypto_4h_fast.toml`: tuned defaults suitable for 4h crypto data.

Improvement loop
- `scripts/auto_improve.py` supports uv, passthrough flags, and a capacity sweep:
  - `--sweep-capacity` to grid‑sweep `fresh_rate`, `pop_size`, `hof_per_gen`.
  - `--seeds 42,7,99` to evaluate across multiple seeds.
  - Summary CSV is written under `pipeline_runs_cs/sweeps/`.

Minimal UI flow
1) Start a run via CLI (spawn `uv run run_pipeline.py ...`), or optionally call a thin Python wrapper.
2) Poll for the presence of `run_dir/SUMMARY.json` and `backtest_portfolio_csvs/backtest_summary_topN.csv`.
3) Render the top rows (Sharpe, AnnReturn, MaxDD, Turnover, Ops, AlphaID) and link to the timeseries PNGs.
4) Offer a “rerun with tweaks” panel that maps directly to CLI flags (UI → CLI).

Notes
- `SUMMARY.json` now includes `schema_version` and a `best_metrics` block for the top‑Sharpe alpha.
- The filesystem layout and filenames are designed to be stable across runs.
- If you need an HTTP API, a minimal FastAPI app can import `utils/run_index` and forward these artefacts.

HTTP API (iterative dashboard server)
- `POST /api/run` → start an `auto_improve` job, returns `job_id`
- `GET /api/events/{job_id}` → SSE stream (status, progress, diag, score)
- `POST /api/stop/{job_id}` → stop a running job
- `GET /api/last-run` → latest run_dir + best Sharpe
- `GET /api/diagnostics?run_dir=...` → returns `diagnostics.json`
- `GET /api/backtest-summary?run_dir=...` → returns backtest summary JSON (list of rows)
- `GET /api/alpha-timeseries?run_dir=...&alpha_id=Alpha_01` → returns per‑alpha timeseries JSON for plotting
