# Alpha Evolve Orientation

This guide orients new contributors (including automated code-generation models) so they can quickly align with the project goals and constraints.

## Mission Objective: Alpha Evolution
- **Core goal:** Discover and iteratively improve quantitative trading signals (alphas) that outperform baselines across multiple markets.
- **Success metrics:** Sharpe ratio, drawdown discipline, diversification, and robustness across repeated backtests and diagnostics.
- **Strategic intent:** Maintain a living research pipeline that continuously surfaces better alphas while capturing experiment provenance for review.

## How We Pursue the Objective
- **Evolutionary pipeline (`run_pipeline.py` & companions):** Runs generation-based searches, backtests candidates, and writes structured artefacts under `pipeline_runs_cs/` for inspection.
- **Auto-improve (`scripts/auto_improve.py`):** Targets focused fine-tuning sweeps when a promising alpha or configuration needs polishing.
- **Experiment artefacts:** JSON summaries, CSV backtest outputs, and meta snapshots capture each run’s configuration and outcomes for reproducibility.

## Operator-Facing Dashboard (New Django Stack)
- **ASGI app (`scripts/dashboard_server/app.py`):** Django replaces FastAPI to streamline middleware reuse and integrate with existing admin-friendly tooling.
- **Routes:** `/api/*` endpoints expose run orchestration, progress streaming, and asset retrieval; `/ui/*` serves the built dashboard SPA.
- **Job plumbing (`scripts/dashboard_server/jobs.py`):** Manages subprocess execution, streaming logs, and lifecycle controls for pipeline and auto-improve jobs.
- **Test coverage:** `tests/test_dashboard_routes.py` ensures the Django endpoints keep parity with the former FastAPI behavior.

## Environment & Tooling Expectations
- **Python 3.12+, `uv` workflow:** Dependencies pinned in `pyproject.toml`/`requirements.txt`; prefer `uv run` or the project-provided virtualenv.
- **Data layout assumptions:** `pipeline_runs_cs/`, `configs/`, and `self_evolution/` contain canonical artefacts used by APIs and tests.
- **Logging:** Background jobs emit structured progress lines (DIAG/PROGRESS) consumed by the dashboard for live updates.

## Contribution Tips for New Models
- Start with existing tests to understand API contracts; extend them when adding endpoints or run modes.
- Preserve CLI parity—humans still launch scripts directly via `uv run` and expect consistent arguments.
- Surface configuration changes through the dashboard JSON responses so the UI remains authoritative.
- When introducing new alpha evaluation strategies, document metrics and storage formats alongside code to keep downstream consumers in sync.

Use this overview as a compass when drafting changes or evaluating regressions—every contribution should strengthen the alpha evolution mission while respecting the surrounding orchestration and tooling.
