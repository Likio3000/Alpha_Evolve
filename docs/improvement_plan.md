# Alpha Evolve Improvement Plan

This plan outlines key opportunities to strengthen Alpha Evolve across its research pipeline, product experience, and operational tooling. Each area lists the current gap, the impact we expect, and actionable next steps ordered roughly from highest to lowest priority.

## 1. Evolution & Research Pipeline
- **Current gaps:** Fitness still leans on a single blended score, style-factor penalties are coarse, and MAP-Elites integration is early-stage.
- **Why it matters:** Richer objectives and diversity pressure raise the odds of discovering resilient alphas that stay performant out-of-sample.
- **Actions:**
  1. Run ablation studies on the multi-objective selection heuristics to quantify trade-offs between turnover, drawdown, and factor exposure.
  2. Promote the quality-diversity archive to a first-class citizen: persist grid metadata, expose archive sampling controls, and track archive coverage over time.
  3. Expand fitness aggregation to include rolling horizon windows (e.g., 1D/5D/20D) with user-tunable weights and confidence penalties for high variance horizons.
  4. Introduce automated hyperparameter sweeps for evaluation horizons, MAP-Elites bins, and selection pressures using the self-evolution controller.
  5. Publish benchmark suites (baseline configs + seeds) so contributors can compare new evolutionary operators against a stable reference.

## 2. Data & Feature Engineering
- **Current gaps:** Feature inputs focus on equity OHLC data, sector mapping is optional, and alternative data integrations are manual.
- **Why it matters:** Broader, better-curated features unlock new alpha families and reduce overfitting to narrow regimes.
- **Actions:**
  1. Add ingestion utilities for common factor datasets (e.g., Fama-French, macro indicators) and document schemas in `docs/reference`.
  2. Ship validators that confirm data completeness, handle missing values, and flag outliers before runs start.
  3. Implement a feature provenance registry that records which transformations contributed to successful programs.
  4. Build a plug-in system for external data feeds (crypto, rates, on-chain metrics) with caching and permission-aware secrets handling.
  5. Provide synthetic data generators to stress-test the pipeline under known regimes and validate neutrality penalties.

## 3. Backtesting & Risk Controls
- **Current gaps:** Stress testing is rudimentary, transaction cost modelling is basic, and portfolio constraints are largely static.
- **Why it matters:** Robust out-of-sample performance hinges on realistic frictions and scenario analysis.
- **Actions:**
  1. Extend the backtest engine with liquidity-aware slippage models, borrow costs, and optional shorting limits.
  2. Add Monte Carlo scenario replays (block bootstraps, regime shifts) and integrate their metrics into evolution penalties.
  3. Surface margin usage, exposure by sector, and drawdown heatmaps in both pipeline outputs and the dashboard.
  4. Automate comparison of evolved alphas against baseline strategies (e.g., equal-weight sector baskets) with significance testing.
  5. Gate Hall-of-Fame promotion on passing configurable robustness checks (turnover caps, drawdown ceilings, sensitivity thresholds).

## 4. Automation & Self-Evolution Loop
- **Current gaps:** The self-evolution agent requires manual approvals and lacks adaptive exploration policies.
- **Why it matters:** A smooth human-in-the-loop automation path accelerates research velocity while keeping risk controls in place.
- **Actions:**
  1. Implement policy templates (bayesian optimization, evolutionary strategies) that propose parameter sets without manual curation.
  2. Enable batched run scheduling with prioritisation queues so the controller can launch multiple experiments concurrently.
  3. Track parameter-response history in a dedicated datastore and expose model fit diagnostics (e.g., surrogate model residuals).
  4. Allow guardrails that automatically halt campaigns when risk metrics deteriorate or compute budgets are exceeded.
  5. Generate shareable experiment summaries (config + highlights) that can be copy/pasted into collaboration tools.

## 5. Dashboard & User Experience
- **Current gaps:** Responsiveness under load needs work, advanced archive visualisations are limited, and configuration discoverability is fragmented.
- **Why it matters:** The dashboard is now the primary control surface; usability directly influences adoption and developer efficiency.
- **Actions:**
  1. Profile and optimise server endpoints that block UI interactions during active runs; consider websocket streaming for long polls.
  2. Add archive drill-down views (heatmaps, slice filters) and make charts linkable for deep dives.
  3. Expand configuration tooling with preset comparisons, inline documentation, and diff views for saved snapshots.
  4. Improve accessibility (keyboard navigation, high-contrast themes) and responsive layouts across breakpoints.
  5. Instrument client performance (Core Web Vitals) and feed metrics into CI dashboards for regression detection.

## 6. Developer Productivity & Tooling
- **Current gaps:** Type coverage is growing but not enforced everywhere; end-to-end tests rely on manual orchestration; dependency updates are ad-hoc.
- **Why it matters:** High-quality tooling keeps the codebase healthy as contributors extend the system.
- **Actions:**
  1. Enforce mypy in CI with strict optional checks on critical modules and add type hints to evolution/backtest components.
  2. Create smoke-test fixtures that spin up the dashboard server, seed mock data, and validate key API flows automatically.
  3. Adopt pre-commit hooks for formatting, linting, and TOML schema validation.
  4. Schedule dependency update automation (e.g., Renovate) with pinned reproducible builds via `uv lock` regeneration.
  5. Document contributor onboarding paths, including local Docker/devcontainer environments and profiling guides.

## 7. Observability & Operations
- **Current gaps:** Run metadata is scattered, alerting is manual, and resource usage is opaque during heavy sessions.
- **Why it matters:** Reliable monitoring shortens incident response and gives confidence in unattended evolution loops.
- **Actions:**
  1. Centralise run telemetry (metrics, logs, configs) into a searchable store (e.g., SQLite/Parquet index + S3 backend).
  2. Instrument Prometheus/Grafana dashboards for CPU, GPU, and queue health; expose alerts for stalled runs or failed tasks.
  3. Record lineage between configs, runs, and resulting alphas to support reproducibility audits.
  4. Provide tooling to snapshot and restore environments (data, configs, checkpoints) for disaster recovery.
  5. Establish rotation schedules and escalation paths for on-call coverage when running long campaigns.

## 8. Documentation & Community
- **Current gaps:** The walkthrough has been expanded, but living tutorials, API references, and contribution guidelines can go further.
- **Why it matters:** Good documentation scales knowledge transfer and invites community participation.
- **Actions:**
  1. Publish scenario-based tutorials (e.g., "from raw CSV to live dashboard analysis") using executable notebooks.
  2. Auto-generate API docs for REST endpoints and Python modules via Sphinx or MkDocs.
  3. Maintain a public roadmap and changelog that maps to GitHub issues to encourage external collaboration.
  4. Launch community office hours or livestreams where maintainers review contributor proposals and recent findings.
  5. Create a gallery of notable evolved alphas with commentary on their behaviours, guardrails, and deployment considerations.

---

Tracking this plan in GitHub projects with owners and timelines will keep progress visible. Revisit priorities quarterly to reflect new research findings or production learnings.
