# Agent Workflow Notes

This project runs inside a constrained sandbox where Python's `multiprocessing` primitives (notably `SemLock`) are unavailable. Full `pytest` runs will block once they hit tests that spawn multiprocessing workers. To keep feedback fast and avoid hangs:

## Local (sandbox) test loop
- Use `./scripts/run_tests_sandbox.sh [pytest args...]`.
- The script activates the virtualenv, exports `SKIP_MP_TESTS=1`, and runs pytest under a short `timeout` (defaults to `10s`).
- Multiprocessing heavy tests (currently all of `tests/test_dashboard_routes.py`) are skipped automatically whenever `SKIP_MP_TESTS=1`.
- Pass additional selectors when you only need a slice, e.g.:
  - `./scripts/run_tests_sandbox.sh -k 'alpha_timeseries'`
  - `PYTEST_TIMEOUT=30 ./scripts/run_tests_sandbox.sh tests/test_evaluation_logic.py`

## Full validation outside the sandbox
- Ask the user to run `uv run pytest` (or `pytest`) locally when changes touch multiprocessing code paths or dashboard server wiring.
- Specifically call out `tests/test_dashboard_routes.py` so they know the multiprocessing coverage still runs on their machine.

## When adding new tests
- If they depend on `multiprocessing`, guard them with a skip similar to the existing pattern so `SKIP_MP_TESTS=1` avoids hanging the sandbox.
- Prefer to mock out process spawning in unit tests where possible to keep the sandbox-friendly suite broad.

## Previewing the dashboard UI from artefacts
- The latest UI capture lives under the `artifacts/now_ui/` slot (`backtest-analysis.png`, `pipeline-controls.png`, `settings-presets.png`, plus `dashboard-server.log`).
- From the CLI you can inspect the slot with `ls artifacts/now_ui` and open the PNGs in your viewer of choice (`xdg-open artifacts/now_ui/pipeline-controls.png` on Linux).
- If you need fresh shots, rerun `npm run capture:screens` inside `dashboard-ui/`; the helper rotates previous captures into `artifacts/past_ui/` and `artifacts/past_ui2/` automatically.
- Share these PNGs with other agents or humans when you need to illustrate the current dashboard state without rebuilding the app.
