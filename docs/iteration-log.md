# Iteration Log

This log tracks the rolling artefact slots under `artifacts/now_ui`,
`artifacts/past_ui`, and `artifacts/past_ui2`. Each slot holds the latest UI
PNG captures plus a copy of the backend server log so the full context is easy
to ingest by agents.

## Refresh Instructions
- Run `python3 scripts/dev/run_iteration.py` for the full start → build → capture → stop loop.
- For manual operation:
  - Start the backend with `python3 scripts/dev/server_manager.py start --log-file logs/iteration/dashboard.log`.
  - Rebuild the UI if necessary (`npm run build`).
- Capture screenshots with `npm run capture:screens` (they land in `artifacts/now_ui/`).
- Copy the freshly written backend log (`logs/iteration/dashboard.log`) into the same slot if you skip the helper.
- The helper script automatically rotates `now_ui → past_ui → past_ui2` on each run.

## Artefact Slots
| Slot | Backtest Analysis | Pipeline Controls | Settings | Backend Log | Notes |
|------|-------------------|-------------------|----------|-------------|-------|
| now_ui | `artifacts/now_ui/backtest-analysis.png` | `artifacts/now_ui/pipeline-controls.png` | `artifacts/now_ui/settings-presets.png` | `artifacts/now_ui/dashboard-server.log` | Most recent capture. |
| past_ui | `artifacts/past_ui/backtest-analysis.png` | `artifacts/past_ui/pipeline-controls.png` | `artifacts/past_ui/settings-presets.png` | `artifacts/past_ui/dashboard-server.log` | Previous capture. |
| past_ui2 | `artifacts/past_ui2/backtest-analysis.png` | `artifacts/past_ui2/pipeline-controls.png` | `artifacts/past_ui2/settings-presets.png` | `artifacts/past_ui2/dashboard-server.log` | Second previous capture (rotated on the next run). |
