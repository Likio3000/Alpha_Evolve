# Iteration Log

This log tracks the two retained artefact slots: `artifacts/latest` (most recent
run) and `artifacts/previous` (immediately prior run). Each slot contains
`logs/` and `screenshots/` subdirectories populated by
`python3 scripts/dev/run_iteration.py` or the equivalent manual steps.

## Refresh Instructions
- Run `python3 scripts/dev/run_iteration.py` for the full start → build → capture → stop loop.
- For manual operation:
  - Start the backend with `python3 scripts/dev/server_manager.py start --log-file artifacts/latest/logs/dashboard.log`.
  - Rebuild the UI if necessary (`npm run build`).
  - Capture screenshots with `npm run capture:screens` (they land in `artifacts/latest/screenshots/`).
- The helper script automatically rotates `latest` to `previous` and prunes any older runs.

## Artefact Slots
| Slot | Overview Screenshot | Settings Screenshot | Manifest | Notes |
|------|---------------------|---------------------|----------|-------|
| latest | `artifacts/latest/screenshots/pipeline-overview.png` | `artifacts/latest/screenshots/settings-presets.png` | `artifacts/latest/screenshots/manifest.json` | Populated by the most recent iteration run. |
| previous | `artifacts/previous/screenshots/pipeline-overview.png` | `artifacts/previous/screenshots/settings-presets.png` | `artifacts/previous/screenshots/manifest.json` | Automatically rotated from the run before the most recent. |

Logs live alongside each slot at `artifacts/<slot>/logs/dashboard.log`.
