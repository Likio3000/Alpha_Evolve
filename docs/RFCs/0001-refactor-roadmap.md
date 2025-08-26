# RFC 0001 – Alpha Evolve Refactor Roadmap

Status: In Progress (A: complete, B: complete, C: mostly complete, D: started)
Authors: Core
Created: 2025-08-26
Last Updated: 2025-08-26

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

3.1 Data lifecycle (decouple globals) — PARTIAL DONE
- Introduce `DataBundle` everywhere: return and thread the object instead of module globals.
- Provide a `Loader` facade that materializes `DataBundle` and exposes diagnostics.
- Keep a tiny adapter layer to maintain current `evolution_components.data_handling` API for backward compatibility during transition.
Implementation:
- `DataBundle` standardized in `utils.data_loading_common`.
- `EvalContext` introduced and now threaded through evaluation and evolution (`evolve_with_context`).
- Per-worker contexts in multiprocessing to avoid global state in workers.
- Module globals remain for compatibility; further reductions tracked under Phase C remaining work.

3.2 Single source of truth for loading/alignment — DONE
- Consolidate all alignment/pruning/`ret_fwd` logic into `utils/data_loading_common.py` (already started).
- Remove residual duplication in `backtesting_components/data_handling_bt.py` and `evolution_components/data_handling.py` by delegating completely.
- Parameterize “required length” semantics for evolution vs backtest (include/exclude lag) centrally.
Implementation:
- Both loaders delegate to `align_and_prune`; timestamp parsing hardened; concise alignment summary logging added.
- Added alignment cache (Phase D) keyed by data-dir metadata + parameters to avoid redundant work.

3.3 Exceptions over exits — DONE (for libraries)
- Define exception hierarchy under `utils.errors` (`DataLoadError`, `ConfigError`, `RuntimeAbort`, etc.).
- Replace remaining `sys.exit` in libraries with exceptions; convert tests accordingly.
- Restrict `sys.exit` to CLI entrypoints only.
Implementation:
- `DataLoadError`, `BacktestError`, `ConfigError` under `utils.errors`.
- Libraries raise exceptions; CLI entrypoints handle and exit.

3.4 CLI normalization — DONE
- Generate CLI from dataclasses (inheritance‑aware): a helper that builds `argparse` parsers from dataclass fields with choices, defaults, aliases.
- Standardize flag names across tools (e.g., `--data_dir`, `--max_lookback_data_option`, `--eval_lag`).
- Maintain legacy aliases for one release with deprecation notices.
Implementation:
- `utils.cli.add_dataclass_args` (inheritance‑aware). Boolean `--no-<flag>` added for True defaults.
- Normalized flags across tools. Legacy short aliases are not accepted anymore (tests enforce).

3.5 Provenance & metadata — DONE
- Persist resolved configs, pruned symbols, overlap windows, seed, and git SHA in `run_dir/meta/` (partially done).
- Add per‑generation diagnostics snapshot (top N programs with metrics, ramp weights, cache stats).
- Produce a compact run `SUMMARY.json` with key metrics and file pointers.
Implementation:
- `run_pipeline` persists `evolution_config.json`, `backtest_config.json`, `run_metadata.json`, `SUMMARY.json`, and `meta/data_alignment.json`.
- Centralized diagnostics saved to `diagnostics.json` with per‑gen stats; HOF snapshot per generation persisted as `meta/hof_gen_<N>.json`.

3.6 Diagnostics & plotting — DONE (core), PLOTTERS unchanged
- Centralize diagnostic collection in an event hub (`utils/diagnostics.py`) used by eval and HOF.
- Standardize plot generation inputs/outputs; tolerate absence of plotting deps.
Implementation:
- `utils.diagnostics` used across evaluation/evolution; run pipeline persists diagnostics.
- Plot scripts remain as optional utilities; pipeline attempts plotting if available.

3.7 Testing strategy — DONE (initial)
- Add an evolution→backtest smoke test with fixed seed and minimal data (already added programmatic BT test).
- Property tests for loader (duplicates, gaps, timezone, min length).
- Golden tests for CLI parse→config mapping.
Implementation:
- Added golden tests for CLI/dataclass mapping, config layering precedence, print-config output; existing integration/backtest tests pass.

3.8 Performance & parallelism — PARTIAL DONE
- Introduce a per‑worker `EvalContext` that holds read‑only `DataBundle` references; consider shared memory for aligned arrays.
- Avoid repeated reindexing by precomputing index alignment once; cache via content hashes.
Implementation:
- Per‑worker `EvalContext` with matrix precomputation for vector features.
- Alignment cache to avoid repeated alignment work.
- Remaining: shared memory/memmap for large matrices across workers.

3.9 Config layering — DONE
- Support file‑based configs (YAML/TOML) with env/CLI overrides (precedence: file < env < CLI).
- Provide presets for SP500 vs Crypto in `configs/`.
Implementation:
- `utils/config_layering` with TOML/YAML loaders, env overrides (`AE_`, `AE_EVO_`, `AE_BT_`), and CLI precedence; `--config` in both tools.
- `--print-config` dumps resolved config as JSON.

3.10 API hygiene — PARTIAL DONE
- Define public API surface (modules, functions, dataclasses) and mark internals.
- Add deprecation shims for renamed functions, with removal timeline.
Implementation:
- Introduced `evolve_with_context` as context‑first API; kept `evolve` as compatibility wrapper.
- Remaining: deprecation notices for globals-based accessors once consumers migrate.

3.11 Backtest consistency — PARTIAL DONE
- Align naming and defaults between evolution and backtest for scale, risk controls, and stops.
- Validate cross‑module assumptions (e.g., `eval_lag` constraints for stops) at parse time.
Implementation:
- Stop‑loss validated to require `eval_lag==1`; flags normalized; risk controls available on backtest.

3.12 Data quality & schema — PARTIAL DONE
- Formalize CSV schema and validations (time monotonicity, duplicates, OHLCV present, timezone handling).
- Optional exchange calendars for daily vs 4h; configurable gap filling policy.
Implementation:
- Hardened timestamp parsing (epoch seconds and ISO8601); duplicate index dedup; OHLC enforcement.
- Remaining: calendar integration and policy configs.

3.13 Hall of Fame improvements — DONE (core)
- Persist HOF entries (program + metrics + preds digest) per gen for auditability.
- Configurable correlation metric (rank, Pearson) and exact‑match fast path (already added).
- Deterministic tie‑breaking and duplicate policy clarified.
Implementation:
- HOF correlation checks (cutoff/weight), exact‑match fast path, deterministic updates, per‑gen snapshots in diagnostics and files.

3.14 Logging UX — DONE
- Central logger setup with per‑module levels and consistent prefixes.
- Summarize kept vs dropped symbols early with counts.
Implementation:
- TQDM‑friendly logging with color; concise alignment summaries on load.

3.15 Scripts & packaging — DONE
- Consolidate recommended scripts into a single `recommended_pipeline.sh` with presets.
- Package project (`pyproject.toml`) with console entry points for pipeline/backtest.
- Pre‑commit hooks: formatting, linting, type checks (opt‑in).
Implementation:
- Project exposes console entries (`alpha-evolve-pipeline`, `alpha-evolve-backtest`).
- Legacy shell scripts removed; configs in `configs/` recommended.
- Pre‑commit config added for Ruff/Black.

## 4) Phased Plan

- Phase A (surgical): COMPLETE
  - Exceptions in libraries; single alignment path; standardized metadata persisted.

- Phase B (ergonomics): COMPLETE
  - CLI from dataclasses; normalized flags; config layering (file < env < CLI); presets in `configs/`.

- Phase C (architecture): MOSTLY COMPLETE
  - `EvalContext` threaded end‑to‑end; per‑worker contexts.
  - Remaining: eliminate globals from evolution/data_handling call sites; deprecations and migration guides.

- Phase D (perf + DX): STARTED
  - Alignment cache; precomputed feature matrices; pre‑commit hooks.
  - Remaining: shared memory/memmap for matrices; CI polish.

## 5) Risks & Mitigations

- Risk: Behavior drift in loaders.
  - Mitigation: Snapshot tests and explicit semantic flags (include_lag) already introduced.
- Risk: CLI churn.
  - Mitigation: Maintain legacy aliases for one release with warnings.
- Risk: Parallel data sharing complexities.
  - Mitigation: Start with read‑only contexts and copy‑on‑write patterns.

## 6) Acceptance Criteria

- No `sys.exit` in libraries; libraries raise exceptions (DONE).
- Single alignment code path; evolution/backtest semantics covered via flags (DONE).
- CLI parsers generated from dataclasses; flags consistent across tools; legacy aliases removed (DONE).
- Runs persist standardized metadata and diagnostics; tests include evo→bt smoke path (DONE).
- Context‑first API available; globals retained temporarily for compatibility (PARTIAL).

## 7) Appendices

### A. Flag normalization map (final)

- `--data_dir` (no `--data` alias)
- `--max_lookback_data_option` (no `--data_alignment_strategy` alias)
- `--eval_lag` (no `--lag` alias)

### B. Exception hierarchy (current)

```text
utils.errors.AlphaEvolveError
 ├─ DataLoadError
 ├─ ConfigError
 ├─ EvaluationError
 └─ BacktestError
```

### C. Phase C remaining work

- Remove/depcreate globals usage in evolution/data modules; prefer explicit contexts.
- Add deprecation warnings and migration notes for any globals-based accessors.

### D. Phase D remaining work

- Shared memory/memmap for precomputed matrices across workers.
- CI polish (optional pre-commit in CI), performance profiling on larger datasets.
