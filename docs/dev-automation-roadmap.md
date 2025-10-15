# Alpha Evolve Iteration Automation Roadmap

## Goals
- Allow the coding agent to control the dashboard backend without manual terminal interaction.
- Provide deterministic frontend snapshots (Pipeline Overview, Settings & Presets) for every iteration.
- Capture artefacts (logs, screenshots, metadata) so progress is traceable and reproducible.

## Deliverables
1. **Server Manager CLI** – launch/stop/restart the Django‑Uvicorn backend, tail logs, and optionally auto‑reload on source changes.
2. **Screenshot Harness** – headless browser script that navigates the dashboard, captures required views, and saves them with deterministic filenames.
3. **Artefact Pipeline** – convention for storing rolling screenshot slots so the latest iteration is easy to locate.
4. **Developer Docs & Convenience Targets** – concise usage docs (this file plus command references) and a top-level helper command (e.g. `make iterate` or `just iterate`) that sequences the workflow.

## Implementation Plan

### 1. Server Manager CLI (`scripts/dev/server_manager.py`)
- Build a Python CLI with subcommands:
  - `start` – boot `scripts/run_dashboard.py` as a subprocess, redirect combined stdout/stderr to `logs/server/latest.log`, and write a PID file.
  - `stop` – terminate the tracked process gracefully; fall back to SIGKILL if unresponsive.
  - `restart` – convenience wrapper around `stop` + `start`.
  - `status` – show running state and log tail.
  - `tail` – follow the current log file.
- Add `--watch` flag that uses `watchfiles` to monitor `scripts/dashboard_server`, `dashboard-ui/dist`, and relevant config directories; on change, auto-restart the backend.
- Ensure the CLI respects `AE_PIPELINE_DIR` and other key environment variables by loading `.env` if present.

### 2. Screenshot Harness (`dashboard-ui/scripts/captureScreenshots.mjs`)
- Use Playwright (Chromium) to:
  1. Confirm the backend is reachable (configurable base URL, default `http://127.0.0.1:8000/ui`).
  2. Capture `Pipeline Overview` (default landing tab).
  3. Switch to `Settings & Presets`, wait for data panels, capture screenshot.
  4. Save images to `artifacts/now_ui/` (the script expects the directory to be prepared in advance by the iteration runner).
- Add stable selectors by introducing `data-test` attributes in `dashboard-ui/src/modules/components/HeaderNav.tsx` and other key elements rendered in each view.
- Wire an npm script: `"capture:screens"` that runs the Node helper to gather screenshots.

### 3. Artefact & Metadata Workflow
- Maintain three rolling slots under `artifacts/`:
  - `artifacts/now_ui/` – current capture pair (PNG-only).
  - `artifacts/past_ui/` – rotated from the previous run.
  - `artifacts/past_ui2/` – rotated from two runs ago.
- The iteration runner handles rotation (`now_ui` → `past_ui` → `past_ui2`) and removes anything older than one generation.
- `docs/iteration-log.md` documents the slot layout and usage reminders.

### 4. Convenience Entrypoints
- `scripts/dev/run_iteration.py` orchestrates the default loop:
  1. Rotate artefact slots (`now_ui` → `past_ui` → `past_ui2`) and stage `artifacts/now_ui/`.
  2. Start the backend via `server_manager.py` (unless `--reuse-server` is passed).
  3. Wait for `http://127.0.0.1:8000/ui` (configurable) to become responsive.
  4. Run `npm run build` (skip with `--skip-build`) and `npm run capture:screens`.
  5. Stop the backend after capture completes.
- Future enhancement: wrap this script in a `make/just iterate` target once additional automation steps are defined.

## Open Questions / Follow-ups
- Should backend tests be triggered automatically before screenshot capture? (`uv run -m pytest` vs targeted suites.)
- Define retention policy for old artefacts (`scripts/cleanup_runs.py` style helper?).
- Decide whether to integrate UI diffing (compare screenshots vs previous iteration) now or later.
