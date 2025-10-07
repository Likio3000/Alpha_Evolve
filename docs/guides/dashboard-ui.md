# Dashboard UI Workflow

The dashboard server (`uv run scripts/run_dashboard.py`) serves a static single-page app from `dashboard-ui/dist/`. Follow these steps when you need to customise or refresh the front-end bundle.

## When You Only Need the Bundled UI
No action required. The repository ships with a pre-built dashboard that covers:
- Launching and monitoring pipeline runs
- Viewing backtest tables and alpha equity curves
- Labelling and downloading run artefacts

Keep the bundle under version control so other contributors get the same assets.

## Rebuilding the Private Front-End
1. Clone or update the internal React/Vite (or preferred stack) source project.
2. Install the Node.js toolchain (Node 18+ recommended) and the package manager your project uses (`pnpm`, `yarn`, or `npm`).
3. From the UI project root run the build pipeline, for example:

   ```bash
   pnpm install
   pnpm build
   # or npm install && npm run build
   ```

4. Copy the generated `dist/` output into this repository’s `dashboard-ui/dist/` directory, overwriting existing files.

Commit the result if you want peers to receive the update. Otherwise add the files to `.git/info/exclude` locally.

## Running the UI in Dev Mode
- Keep the Python dashboard running (`uv run scripts/run_dashboard.py`) and point it at your run directory via `AE_PIPELINE_DIR`.
- Start the front-end dev server (e.g. `pnpm dev`) and configure its proxy to forward API calls to `http://127.0.0.1:8000`.
- Adjust CORS rules in the UI dev server if you are testing from a different origin.

The Python backend exposes configuration metadata (`/ui-meta/*`, `/api/config/*`) so the UI can render dynamic forms without duplicating schemas. Refer to `scripts/dashboard_server/ui_meta.py` and the Pydantic models in `scripts/dashboard_server/models.py` when adding new controls.

## Asset Downloads and Telemetry
The dashboard uses:
- `/api/job-status/<job_id>` and `/api/job-log/<job_id>` for live run updates.
- `/api/backtest-summary` and `/api/alpha-timeseries` for table/chart data.
- `/api/run-asset` to stream files such as PNGs straight from the run directory.

If you add new artefacts or routes, update the UI bundle along with the documentation so consumers stay in sync.
