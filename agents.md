# Agents Backlog

- [x] Add regression tests for `scripts/dashboard_server/helpers.resolve_latest_run_dir` and `/api/last-run` to prevent future path normalization bugs.
- [x] Normalize the `pipeline_runs_cs/LATEST` writer in `run_pipeline.py` to emit project-relative paths (plus migration script) so older runs stay compatible.
- [x] Expose configurable pipeline output directory via CLI/env and document safe storage locations outside the repo to avoid accidental large check-ins.
- [x] Add FastAPI integration tests (TestClient) covering runs/backtest endpoints and expected 4xx/5xx paths.
- [x] Provide build instructions (or scripts) for the advanced Node dashboard UI and ensure the Python server serves the latest assets.
- [x] Automate housekeeping for `pipeline_runs_cs` (retention policy, disk usage warnings) to keep long-running experiments tidy.
- [x] Expand docs with an end-to-end “first run” walkthrough including data prep, dashboard usage, and interpreting diagnostics/backtest outputs.
- [x] Introduce mypy (or `uv run mypy`) and enforce via CI so config/dataclass changes fail fast on type regressions.
- [x] Incorporate style-factor neutralization (market/size/liquidity) penalties into `evaluation_logic.py` to favour idiosyncratic alphas.
- [x] Score programs across multiple holding horizons and blend those metrics into both evolution fitness and backtests.
- [x] Prototype a quality-diversity archive (MAP-Elites style) to replace/augment the current Hall of Fame heuristics.
- [x] Instrument diagnostics with factor exposure traces, per-horizon stats, feature coverage, and MAP-Elites summaries; surface via dashboard APIs.
- [ ] Enable/extend multi-objective selection so turnover, drawdown proxies, and factor exposures are considered alongside fitness.
- [ ] Expand feature inputs (volatility spreads, cross-asset signals, on-chain metrics) and surface regime-aware diagnostics.
- [ ] Integrate richer transaction-cost and stress backtests, feeding robustness penalties back into evolution and dashboards.
- [ ] Update dashboard UI to expose new pipeline knobs (`factor_penalty_*`, `evaluation_horizons`, QD archive controls).
- [ ] Extend dashboard views to display generation diagnostics (`factor_exposure_summary`, `horizon_summary`, `feature_coverage`, `qd_summary`).
