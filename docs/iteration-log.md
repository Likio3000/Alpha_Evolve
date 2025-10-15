# Iteration Log

This log tracks artefacts generated during iterative development passes. Each row
should reference the screenshots captured by `npm run capture:screens` and any
associated backend logs.

## Refresh Instructions
- Start the backend via `python3 scripts/dev/server_manager.py start --log-file artifacts/logs/<timestamp>/dashboard.log`.
- Build the UI bundle if required (`npm run build`).
- Capture screenshots with `npm run capture:screens`. The script emits PNGs under `artifacts/screenshots/<timestamp>/`
  and updates `artifacts/screenshots/latest.json`.
- Copy the resulting artefact paths into the table below and note any relevant context.

## Artefacts
| Timestamp | Overview Screenshot | Settings Screenshot | Manifest | Notes |
|-----------|---------------------|---------------------|----------|-------|
| 2025-10-15T09:35:19Z | `artifacts/screenshots/2025-10-15T09-35-19-671Z/pipeline-overview.png` | `artifacts/screenshots/2025-10-15T09-35-19-671Z/settings-presets.png` | `artifacts/screenshots/2025-10-15T09-35-19-671Z/manifest.json` | Generated via `python3 scripts/dev/run_iteration.py --reuse-server --skip-build` |
