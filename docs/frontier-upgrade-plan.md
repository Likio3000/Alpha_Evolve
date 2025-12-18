# Alpha Evolve Frontier Upgrade Plan

Date: 2025-12-17

## Target
- **Dataset:** Daily S&P 500 workflow (`configs/sp500.toml` + `data_sp500/`).
- **Primary objective:** higher **out-of-sample Sharpe**.
- **Secondary objective:** **low correlation among the final shipped alphas/strategies** (diversification), but never at the expense of large Sharpe losses.

This plan is structured as test-gated milestones. Every milestone is considered “done” only if it passes the listed verification commands and produces the expected artefacts/UI behavior.

## Constraints (current repo reality)
- The sandbox runtime used by agents can lack `multiprocessing` primitives (see `agents.md`). Full multiprocessing coverage must be run on a normal machine, but we will add a sandbox-safe runner so end-to-end validation is still possible here.
- Network access may be restricted; avoid introducing “install from internet” assumptions into scripts.

## Verification Gates (run continuously while building)

### Fast sandbox loop (agent can run)
- `./scripts/run_tests_sandbox.sh`
- `npm --prefix dashboard-ui run lint && npm --prefix dashboard-ui run build`
- `bash scripts/smoke_run.sh` (after we make it sandbox/venv-safe)

### Full local validation (user runs on their machine)
- `uv run pytest` (or `pytest`) with multiprocessing tests enabled
- A real SP500 daily run: `python -m alpha_evolve.cli.pipeline ... --config configs/sp500.toml`

## Roadmap (phased, each phase shippable)

### Phase 1 — “Make it run” reliability (foundation)
**Deliverables**
- Fix bootstrap so `alpha_evolve` is importable in a fresh venv:
  - Update `scripts/setup_env.sh` to install the package (`pip install -e .`) or provide a documented `PYTHONPATH=src` fallback.
- Make `uv` usage sandbox-safe by default (workspace cache dir) and remove hidden reliance on `~/.cache`.
- Add `scripts/verify.sh` (single command) that runs the fast gates above.
- Ensure `scripts/smoke_run.sh` works in a clean environment and produces:
  - `pipeline_runs_cs/LATEST`
  - `run_*/SUMMARY.json`
  - `run_*/meta/*.json`

**Acceptance**
- Fresh clone → `sh scripts/setup_env.sh` → `bash scripts/smoke_run.sh` succeeds.

**Tests**
- Run the full “Fast sandbox loop”.

---

### Phase 2 — Robust job execution in constrained runtimes (backend)
**Deliverables**
- Add a selectable job runner mode for `/api/pipeline/run`:
  - **Thread/subprocess mode** (sandbox-safe, no `multiprocessing` primitives).
  - **Multiprocessing mode** (default on real machines).
- Keep the event model identical across modes:
  - SSE stream (`/api/pipeline/events/<job_id>`) emits structured events: `log`, `progress`, `score`, `gen_summary`, `final`.
  - Persist `meta/ui_context.json` and `meta/gen_summary.jsonl`.
- Add a deterministic cancel/stop behavior for both modes.

**Acceptance**
- Starting jobs, stopping jobs, and streaming progress works in both runner modes.

**Tests**
- Add ASGI integration tests for:
  - job start → health remains responsive
  - SSE emits well-formed frames
  - stop endpoint changes status and terminates work

---

### Phase 3 — Frontier monitoring UX (UI + backend contract)
**Deliverables**
- UI switches to **SSE-first** consumption (EventSource):
  - no regex scraping for Sharpe; use `score` events.
  - reconnect logic: if SSE drops, use `/api/job-activity/<job_id>` snapshot to resume.
  - connection state shown in UI (connected/retrying/stale).
- Add run-forensics surfacing:
  - show `meta/gen_summary.jsonl` as a timeline and allow quick diff between generations.
  - add an artefact browser powered by `/api/run-asset` (CSV/JSON/PNG preview/download).

**Acceptance**
- Near real-time progress updates; refresh/reconnect doesn’t lose state.

**Tests**
- UI build passes.
- Backend tests assert event schema stability (contract tests).

---

### Phase 4 — Evolution math upgrades (Sharpe-first, diversify final set)
This phase changes the *selection math* to improve the probability of finding high-Sharpe alphas while ensuring the final selected set is not redundant.

#### 4A) Sharpe-first, noise-aware selection (reduce “lucky winners”)
**Deliverables**
- Add uncertainty-aware objective variants:
  - probability-of-superiority style scoring (e.g. PSR/DSR-like bounds) for Sharpe proxies / IC on validation.
  - robust aggregation across time slices / folds (median/trimmed mean) to reduce regime overfit.
- Ensure purged CV + embargo is correct and consistently applied when enabled (`cv_k_folds`, `cv_embargo`).
- Make the selection metric explicitly configurable:
  - `selection_metric = auto|phased|ramped|fixed|ic` remains, but with options like `psr`/`lcb` (lower confidence bound) for exploitation phases.

**Acceptance**
- On controlled synthetic cases, uncertainty-aware variants select the more stable signal over a higher-variance lucky one.

**Tests**
- Unit tests for PSR/LCB math and fold splitting invariants (no leakage).

#### 4B) Diversity mechanics during evolution (cheap, Sharpe-preserving)
**Principle:** we don’t forbid correlation during search, but we avoid collapsing the HOF into duplicates.

**Deliverables**
- Track **two correlations**:
  1) **Signal correlation** (cheap): correlation between candidate prediction vectors and HOF prediction vectors (already partially supported via `hof_corr_mode`).
  2) **Return correlation** (meaningful): correlation between backtested daily alpha returns for the top candidates.
- Use correlation mainly as:
  - a tie-breaker / secondary objective in MOEA,
  - and a **gentle** penalty (ramped up late, not early).
- Expand novelty scoring:
  - structural novelty (opcode Jaccard) + behavioral novelty (prediction distance on a probe window).

**Acceptance**
- HOF contains multiple distinct families, not one repeated template.

**Tests**
- Unit tests for correlation metrics and novelty scores (including NaN handling).

#### 4C) Final alpha set selection (where “uncorrelated” is enforced)
**Principle:** enforce diversification at the end, after Sharpe candidates exist.

**Deliverables**
- Add a “final selection” step that chooses K alphas by maximizing a Sharpe-centric objective with a soft correlation penalty, e.g.:
  - maximize `expected_portfolio_sharpe - λ * avg_pairwise_corr`
  - or greedy: pick best Sharpe then add the highest-Sharpe alpha that keeps corr under a threshold; threshold relaxed if no candidates fit.
- Use **return correlation** on the validation/out-of-sample slice for this final step.
- Persist correlation matrices + cluster summaries into run artefacts for UI.

**Default policy (Sharpe > corr)**
- Treat correlation as *soft* (penalty/tie-breaker) until Sharpe is “good enough”; then tighten.
- Prefer a small `λ` and/or a high corr ceiling (e.g. 0.7) that can be lowered once the system consistently finds high Sharpe.

**Acceptance**
- The final selected K alphas are materially less correlated than the raw top-K-by-Sharpe, with minimal Sharpe sacrifice.

**Tests**
- Deterministic toy tests where correlation-aware selection prefers diversified picks when Sharpe is similar.

---

### Phase 5 — Benchmark harness (prove improvements on SP500)
**Deliverables**
- Add a reproducible benchmark runner:
  - quick mode: `sp500_small`, few seeds, short gens (CI/sandbox friendly).
  - full mode: daily SP500, multiple seeds, longer gens (user machine).
- Produce comparable reports:
  - distribution of best Sharpe, time-to-threshold, and correlation stats of final K.
  - machine-readable JSON/CSV for dashboards and diffs.

**Acceptance**
- We can answer “did this math change help?” with numbers, not anecdotes.

**Tests**
- Quick benchmark smoke test runs in the sandbox without hanging.

---

### Phase 6 — Agentic experiment manager (self-evolve + dashboard control)
**Deliverables**
- Upgrade `scripts/self_evolve.py` into a tracked experiment system:
  - SQLite registry: configs, diffs, seeds, metrics, run paths, git SHA, dataset hash.
  - dashboard UI to start/monitor sessions, approve/reject candidates, and replay the best.
- Make optimization explicitly multi-objective with Sharpe dominance:
  - primary objective Sharpe; correlation only influences decisions when Sharpe is close or above a threshold.

**Acceptance**
- One click: run a session, compare candidates, replay the winner, and know exactly what changed.

**Tests**
- DB tests, replay tests, and a 1-iteration end-to-end integration test.

---

### Phase 7 — Productionization (optional, high leverage)
**Deliverables**
- Docker/compose, CI gates (python tests + UI build + quick benchmark).
- Structured logs + `/metrics`.
- Run retention tooling and safety checks.

## Implementation Order (recommended)
1) Phase 1 (unblocks everything).
2) Phase 2 (end-to-end pipeline + SSE tests become possible in constrained runtimes).
3) Phase 3 (observability UX improves iteration speed).
4) Phase 4 (math upgrades with unit tests + benchmark harness hooks).
5) Phase 5 (prove improvements).
6) Phase 6–7 (agentic sessions + hardening).

