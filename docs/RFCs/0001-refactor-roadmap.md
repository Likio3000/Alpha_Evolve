# RFC 0001 – Alpha Evolve Refactor Roadmap

Status: Draft
Authors: Core
Created: 2025-08-26

## 1) Context & Goals

This RFC proposes a sequence of refactors to improve correctness, composability, and developer experience across the evolution and backtesting pipeline without altering core algorithms. The goals are to:

- Unify overlapping logic (data loading/alignment, diagnostics, CLI) into single sources of truth.
- Reduce global state and side effects to make components reusable, testable, and parallel‑friendly.
- Normalize CLI and config handling for consistency between evolution and backtest.
- Strengthen provenance/metadata and diagnostics for reproducibility and debugging.

Non‑goals: Change the algorithmic search space, introduce new model types, or materially alter fitness semantics.

## 2) Pain Points (Today)

- Library modules call `sys.exit`, making composition and tests brittle.
- Duplicate data loaders (evolution vs backtest) drift in features and semantics.
- Module‑level globals for data/features complicate reuse and multi‑run workflows.
- CLI flags differ across tools; inherited dataclass fields weren’t always parsed.
- Sparse provenance: pruned symbols and overlap windows are not uniformly persisted.

## 3) Proposed Refactors

3.1 Data lifecycle (decouple globals)
- Introduce `DataBundle` everywhere: return and thread the object instead of module globals.
- Provide a `Loader` facade that materializes `DataBundle` and exposes diagnostics.
- Keep a tiny adapter layer to maintain current `evolution_components.data_handling` API for backward compatibility during transition.

3.2 Single source of truth for loading/alignment
- Consolidate all alignment/pruning/`ret_fwd` logic into `utils/data_loading_common.py` (already started).
- Remove residual duplication in `backtesting_components/data_handling_bt.py` and `evolution_components/data_handling.py` by delegating completely.
- Parameterize “required length” semantics for evolution vs backtest (include/exclude lag) centrally.

3.3 Exceptions over exits
- Define exception hierarchy under `utils.errors` (`DataLoadError`, `ConfigError`, `RuntimeAbort`, etc.).
- Replace remaining `sys.exit` in libraries with exceptions; convert tests accordingly.
- Restrict `sys.exit` to CLI entrypoints only.

3.4 CLI normalization
- Generate CLI from dataclasses (inheritance‑aware): a helper that builds `argparse` parsers from dataclass fields with choices, defaults, aliases.
- Standardize flag names across tools (e.g., `--data_dir`, `--max_lookback_data_option`, `--eval_lag`).
- Maintain legacy aliases for one release with deprecation notices.

3.5 Provenance & metadata
- Persist resolved configs, pruned symbols, overlap windows, seed, and git SHA in `run_dir/meta/` (partially done).
- Add per‑generation diagnostics snapshot (top N programs with metrics, ramp weights, cache stats).
- Produce a compact run `SUMMARY.json` with key metrics and file pointers.

3.6 Diagnostics & plotting
- Centralize diagnostic collection in an event hub (`utils/diagnostics.py`) used by eval and HOF.
- Standardize plot generation inputs/outputs; tolerate absence of plotting deps.

3.7 Testing strategy
- Add an evolution→backtest smoke test with fixed seed and minimal data (already added programmatic BT test).
- Property tests for loader (duplicates, gaps, timezone, min length).
- Golden tests for CLI parse→config mapping.

3.8 Performance & parallelism
- Introduce a per‑worker `EvalContext` that holds read‑only `DataBundle` references; consider shared memory for aligned arrays.
- Avoid repeated reindexing by precomputing index alignment once; cache via content hashes.

3.9 Config layering
- Support file‑based configs (YAML/TOML) with env/CLI overrides (precedence: file < env < CLI).
- Provide presets for SP500 vs Crypto in `configs/`.

3.10 API hygiene
- Define public API surface (modules, functions, dataclasses) and mark internals.
- Add deprecation shims for renamed functions, with removal timeline.

3.11 Backtest consistency
- Align naming and defaults between evolution and backtest for scale, risk controls, and stops.
- Validate cross‑module assumptions (e.g., `eval_lag` constraints for stops) at parse time.

3.12 Data quality & schema
- Formalize CSV schema and validations (time monotonicity, duplicates, OHLCV present, timezone handling).
- Optional exchange calendars for daily vs 4h; configurable gap filling policy.

3.13 Hall of Fame improvements
- Persist HOF entries (program + metrics + preds digest) per gen for auditability.
- Configurable correlation metric (rank, Pearson) and exact‑match fast path (already added).
- Deterministic tie‑breaking and duplicate policy clarified.

3.14 Logging UX
- Central logger setup with per‑module levels and consistent prefixes.
- Summarize kept vs dropped symbols early with counts.

3.15 Scripts & packaging
- Consolidate recommended scripts into a single `recommended_pipeline.sh` with presets.
- Package project (`pyproject.toml`) with console entry points for pipeline/backtest.
- Pre‑commit hooks: formatting, linting, type checks (opt‑in).

## 4) Phased Plan

- Phase A (surgical):
  - Finish exception sweep; update tests.
  - Complete handoff to `align_and_prune` for both loaders.
  - Persist standardized metadata (`SUMMARY.json`, `data_alignment.json`).

- Phase B (ergonomics):
  - CLI generator from dataclasses; normalize flags + aliases.
  - Add presets and config layering.

- Phase C (architecture):
  - Introduce `DataBundle` threading end‑to‑end; deprecate globals.
  - Add `EvalContext` and worker contexts.

- Phase D (perf + DX):
  - Shared memory or memory‑mapped arrays for aligned data.
  - Pre‑commit/tooling, packaging, and docs.

## 5) Risks & Mitigations

- Risk: Behavior drift in loaders.
  - Mitigation: Snapshot tests and explicit semantic flags (include_lag) already introduced.
- Risk: CLI churn.
  - Mitigation: Maintain legacy aliases for one release with warnings.
- Risk: Parallel data sharing complexities.
  - Mitigation: Start with read‑only contexts and copy‑on‑write patterns.

## 6) Acceptance Criteria

- No `sys.exit` in libraries; all loaders/tests use exceptions.
- Single alignment code path; evolution/backtest semantics covered via flags.
- CLI parsers generated from dataclasses; flags consistent across tools; legacy aliases work.
- Runs persist standardized metadata and diagnostics; tests include an evo→bt smoke path.

## 7) Appendices

### A. Flag normalization map (initial)

- `--data` → `--data_dir` (alias kept)
- `--data_alignment_strategy` → `--max_lookback_data_option` (alias kept)
- `--lag` → `--eval_lag` (alias kept)

### B. Exception hierarchy (sketch)

```text
utils.errors.AlphaEvolveError
 ├─ DataLoadError
 ├─ ConfigError
 ├─ EvaluationError
 └─ BacktestError
```

