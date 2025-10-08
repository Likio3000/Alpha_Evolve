# React Dashboard Parity Backlog

This checklist captures legacy dashboard touches that are still missing from the React + TypeScript rewrite. Each gap references the documented/implemented behaviour in the Python backend alongside the current front-end state.

## Live Run Monitoring

- **SSE event stream unhooked.** The backend exposes `/api/pipeline/events/<job_id>` with `log`, `progress`, `diag`, and `score` payloads (`docs/guides/backend-api.md:62-64`), yet the app continues to poll `/api/job-status` and `/api/job-log` on a timer (`dashboard-ui/src/modules/App.tsx:342-395`). Without subscribing to the stream, the UI cannot render real-time progress or diagnostic cards.
- **Sharpe updates scraped from log text.** Instead of consuming structured `score` events, the client falls back to a regex against the log buffer (`dashboard-ui/src/modules/App.tsx:342-364`). The legacy UI used the structured feed, so this rewrite loses accuracy whenever the log format shifts.
- **Per-generation diagnostics panel still absent.** `diagnostics.json` is meant to drive dashboard charts (`docs/reference/run-artifacts.md:5-9`), but the React app never fetches or visualises it—there is no API call for diagnostics in `dashboard-ui/src/modules/api.ts:34-166`, nor any chart component beyond the alpha equity view.

## Run Detail Inspection

- **Run detail panel not implemented.** The backend ships `/api/run-details` with `SUMMARY.json`, `ui_context`, and meta blobs (`scripts/dashboard_server/routes/runs.py:295-329`), and the React shell attempts to mount `RunDetailsPanel` with those responses (`dashboard-ui/src/modules/App.tsx:24`, `dashboard-ui/src/modules/App.tsx:506-517`). However, `dashboard-ui/src/modules/components/` contains no `RunDetailsPanel` implementation, so the extra data never renders.
- **UI context & pipeline args hidden.** `RunDetails` carries the stored submission payload and resolved CLI args (`scripts/dashboard_server/routes/run_pipeline.py:187-208`), yet without the missing panel the React UI offers no way to inspect them, erasing a parity feature from the legacy UI handoff.
- **Run metadata JSONs inaccessible.** Items such as `evolution_config.json`, `backtest_config.json`, and `data_alignment.json` are bundled into the `run-details` payload (`scripts/dashboard_server/routes/runs.py:313-325`), but no component surfaces them, forcing users back to the filesystem.
- **Run asset downloads stripped out.** The docs note `/api/run-asset` for streaming CSV/PNG artefacts (`docs/guides/dashboard-ui.md:35-40`), yet the React API layer never calls that endpoint (`dashboard-ui/src/modules/api.ts:34-166`), leaving the UI without the previous “download snapshot” affordances.

## Configuration & Documentation

- **UI metadata forms missing.** The backend still serves grouped parameter docs (`scripts/dashboard_server/ui_meta.py:7-148`) and the README explains the UI should render dynamic forms from them (`docs/guides/dashboard-ui.md:33`). The React app does fetch both `/ui-meta/pipeline-params` and `/ui-meta/evolution-params` on mount (`dashboard-ui/src/modules/App.tsx:92-124`), but with no component consuming `pipelineDocs`/`evolutionDocs` the information is discarded.
- **Settings tab is read-only tables.** `SettingsPanel` merely compares defaults to active values (`dashboard-ui/src/modules/components/SettingsPanel.tsx:41-284`), so none of the field-level help, bounds, or pickers defined in `ParamMetaItem` (`dashboard-ui/src/modules/types.ts:118-137`) reach the user the way they did in the previous UI.
- **Preset choices unused.** `/api/config/defaults` returns curated choice lists (`docs/guides/backend-api.md:55-58` and `dashboard-ui/src/modules/types.ts:88-92`), but the React panel neither displays the options nor lets the user switch them inline, resulting in a regression from the metadata-driven forms.

## Launch Controls

- **Override inputs missing.** The API still accepts rich overrides alongside `generations` (`docs/guides/backend-api.md:18-27`), but the React form is limited to dataset, generations, config path, and data dir (`dashboard-ui/src/modules/components/PipelineControls.tsx:28-71`). There is no way to toggle flags such as `bt_top`, `disable_align_cache`, or `no_clean` that were exposed in the old dashboard.
- **Saved config inspection gap.** Even though `SettingsPanel` can load and save presets, there is no preview of the underlying TOML sections beyond raw JSON copy—another nicety the legacy UI surfaced via inline editors tied to the `/ui-meta` schema.

Collecting these gaps in one place should make it easier to drive the parity push for the React dashboard.
