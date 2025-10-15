# Iteration Log

This log tracks the rolling artefact slots under `artifacts/now_ui`,
`artifacts/past_ui`, and `artifacts/past_ui2`. Each slot holds only the current
PNG captures so the latest UI state is easy to ingest by agents.

## Refresh Instructions
- Run `python3 scripts/dev/run_iteration.py` for the full start → build → capture → stop loop.
- For manual operation:
  - Start the backend with `python3 scripts/dev/server_manager.py start --log-file logs/iteration/dashboard.log`.
  - Rebuild the UI if necessary (`npm run build`).
  - Capture screenshots with `npm run capture:screens` (they land in `artifacts/now_ui/`).
- The helper script automatically rotates `now_ui → past_ui → past_ui2` on each run.

## Artefact Slots
| Slot | Overview Screenshot | Settings Screenshot | Notes |
|------|---------------------|---------------------|-------|
| now_ui | `artifacts/now_ui/pipeline-overview.png` | `artifacts/now_ui/settings-presets.png` | Most recent capture. |
| past_ui | `artifacts/past_ui/pipeline-overview.png` | `artifacts/past_ui/settings-presets.png` | Previous capture. |
| past_ui2 | `artifacts/past_ui2/pipeline-overview.png` | `artifacts/past_ui2/settings-presets.png` | Second previous capture (rotated on the next run). |

Logs now live under `logs/iteration/` to keep the artefact slots PNG-only.
