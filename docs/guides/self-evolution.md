# Self-Evolution Sessions

`uv run scripts/self_evolve.py` orchestrates outer-loop experimentation on top of the standard pipeline. It perturbs configuration parameters, evaluates candidates, and stores a full audit trail so humans (or agents) can approve iterations.

## Modes
- **Tracked (default):** stores sessions in a SQLite registry and is designed to be driven from the dashboard Experiments tab.
- **Legacy:** stores approvals in `pending_action.json` inside a session directory (useful for offline/manual workflows).

Select explicitly with `--mode tracked|legacy`.

## Workflow Overview
1. Load the base `EvolutionConfig`/`BacktestConfig` from `--config` (or defaults in `alpha_evolve.config`).
2. Sample parameter updates from a search-space file (`--search-space`).
3. Launch the pipeline with those overrides, collecting metrics from `SUMMARY.json` and backtest artefacts.
4. Record the iteration outcome:
   - tracked mode: into the SQLite registry (plus the normal `pipeline_runs_cs/run_*` artefacts),
   - legacy mode: into a session directory, including pending approvals if `--auto-approve` is disabled.

## Search Space Definition
Pass a JSON or TOML file describing tunable parameters. Formats accepted by `self_evolution.load_search_space`:

```json
{
  "parameters": [
    { "key": "generations", "type": "int_range", "min": 5, "max": 25, "step": 5 },
    { "key": "bt_top", "type": "choice", "values": [5, 10, 20] },
    { "key": "novelty_boost_w", "type": "float_range", "min": 0.0, "max": 2.0, "perturbation": 0.25 }
  ]
}
```

Supported `type` values:
- `choice` – random pick from `values`.
- `bool_toggle` – flip the current boolean setting.
- `float_range` / `int_range` – sample within `[min, max]`, optionally snapping with `step`, `round_to`, or `perturbation` around the current value.

Additional keys: `mutate_probability`, `allow_same`, and `description` (surfaced in briefings).

## Important Flags
- `--mode tracked|legacy` – execution mode (default tracked).
- `--iterations` – number of outer-loop steps (default 5).
- `--objective` / `--minimize` – metric name fetched from `SUMMARY.json` and whether to maximise it (default Sharpe, maximise).
- `--exploration-prob` – probability of exploring vs exploiting current best.
- `--session-root` – where to store session artefacts (defaults to `pipeline_runs_cs/self_evolution/`).
- `--registry-db` – SQLite DB path for tracked mode (defaults to `artifacts/experiments/experiments.sqlite`).
- `--pipeline-output-dir` – overrides the run directory for inner pipeline executions.
- `--auto-approve` – skip manual approval; otherwise poll `pending_action.json` until a human updates it.
- `--pipeline-log-level`, `--pipeline-log-file`, `--disable-align-cache`, `--align-cache-dir` – forwarded to `PipelineOptions`.

## Session Artefacts
Legacy mode session directories contain:
- `history.jsonl` – JSON-per-line history of iterations (success and failure).
- `session_summary.json` – aggregate view (best objective, iterations completed).
- `agent_briefings.jsonl` – prompts describing proposed updates when approval is required.
- `pending_action.json` – approval status for the next iteration.
- `generated_configs/` – TOML snapshots of candidate configs when `persist_generated_configs` is enabled.
- `pipeline_runs/` – per-iteration run directories when `--pipeline-output-dir` is unset.

## Approval Loop
### Tracked mode
When `--auto-approve` is false, each iteration creates a proposal row in the SQLite registry and waits for a decision.
Use the dashboard Experiments tab (recommended) to approve/reject proposals, or update the DB manually.

### Legacy mode
When `--auto-approve` is false, each iteration pauses after generating a candidate. Update `pending_action.json` with:
```json
{
  "status": "approved" | "rejected",
  "notes": "Optional rationale"
}
```
The agent polls at `--approval-poll-interval` seconds until approval is granted or an optional `--approval-timeout` elapses.

## Tips
- Seed with `--config configs/sp500.toml` for reproducible baselines.
- Point `--pipeline-output-dir` and `AE_PIPELINE_DIR` at the same location so the dashboard can visualise inner runs.
- Commit useful search-space definitions under `configs/self_evolution/` and document them here when added.
