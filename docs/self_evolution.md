# Self-Evolution Workflow

This repository includes an autonomous controller that can iteratively run the
`run_pipeline.py` workflow, inspect outcomes, and spawn new configuration
candidates. The goal is to accelerate experimentation by letting an agent drive
self-play style loops: observe results, tweak parameters, and trigger fresh
runs automatically while keeping a human in the loop for approvals.

## Key Components

- `run_pipeline.PipelineOptions` – programmatic wrapper extracted from
  `run_pipeline.py` so the pipeline can be invoked with resolved dataclass
  configurations.
- `self_evolution.agent.SelfEvolutionAgent` – orchestrates the
  iterate→evaluate→mutate cycle with a human approval gate. It supports:
  - configurable search spaces for both evolution/backtest configs and
    pipeline-level toggles (e.g. caching or baseline training);
  - exploration/exploitation control; the agent keeps track of the best run and
    can bias mutations around it;
  - persistence of candidate configs, reasoning briefings, and pending actions
    for auditability;
  - pluggable pipelines so tests can inject mocks or alternate runners.
- `scripts/self_evolve.py` – CLI entrypoint that wires everything together. It
  reads a JSON/TOML search-space description and runs the agent for a specified
  number of iterations.

## Usage Example

1. Describe the parameters you want the agent to manipulate. Example
   `configs/self_evolution/sample_crypto_space.json`:

   ```json
   {
     "parameters": [
       {"key": "evolution.pop_size", "type": "choice", "values": [80, 100, 120]},
       {"key": "evolution.parsimony_penalty", "type": "float_range", "min": 0.001, "max": 0.004, "step": 0.0001, "perturbation": 0.0003},
       {"key": "backtest.scale", "type": "choice", "values": ["madz", "zscore"]},
       {"key": "pipeline.persist_hof_per_gen", "type": "choice", "values": [true, false], "mutate_probability": 0.25}
     ]
   }
   ```

2. Launch the agent:

   ```bash
   uv run scripts/self_evolve.py \
     --config configs/crypto.toml \
     --search-space configs/self_evolution/sample_crypto_space.json \
     --iterations 8 --seed 17 --objective Sharpe
   ```

   Prefer the dashboard? Open the **Self-Play** page (Self-Play button in the
   top bar), fill in the same inputs (search-space, config, iterations, etc.),
   and click **Launch Self-Play** to start a session directly from the UI.

   Each iteration produces a fresh pipeline run (with artefacts in
   `pipeline_runs_cs` unless overridden) and logs metadata under a dedicated
   `pipeline_runs_cs/self_evolution/self_evo_session_*` directory:

   - `history.jsonl` – JSON-lines audit trail listing the parameter tweaks and
     resulting metrics for every iteration.
   - `agent_briefings.jsonl` – per-iteration reasoning (alpha counts, Sharpe
     deltas, plateau detection, suggested next tweaks).
   - `pending_action.json` – live approval state. After each run the agent writes
     its proposed next configuration here and pauses until you edit `status` to
     `approved`, `rejected`, or `stop`.
   - `generated_configs/candidate_XXX.json` – fully materialised configs that
     can be re-run explicitly.
   - `session_summary.json` – quick link to the best recorded run plus pointers
     to the briefing/pending files.

### Approval workflow

By default the agent pauses between runs and waits for you to review the
results:

1. Inspect the latest entry in `agent_briefings.jsonl` (or the UI feed) for
   metrics and commentary.
2. Edit `pending_action.json`, set `status` to `approved`/`rejected`/`stop`, and
   optionally override the proposed config via `approved_candidate`.
3. Save the file. The agent detects the change and either launches the next run
   or ends the session.

For unattended automation pass `--auto-approve` to the CLI (or set
`auto_approve=True` in `AgentConfig`).

The dashboard’s **Self-Play** page mirrors the same data: you can see briefings,
pending state, and action buttons to approve/request tweaks/stop without
touching the JSON files manually. Notes typed into the UI are persisted back to
`pending_action.json` for traceability.

## Extending the Agent

The agent was designed as a scaffold:

- Inject a custom `pipeline_runner` when instantiating `SelfEvolutionAgent` to
  plug in alternative evaluation loops (e.g. UI-driven flows or multi-stage
  pipelines).
- The parameter space understands `choice`, `float_range`, `int_range`,
  and simple boolean toggles. Add new kinds by extending
  `ParameterSpec._generate_value`.
- Session directories are deterministic and git-friendly, so you can study diffs
  or recover any candidate configuration with confidence.

For more complex self-play strategies (e.g. reinforcement learners that modify
code in addition to configs), build on the hooks provided in
`SelfEvolutionAgent`: it already exposes the selected configs, updates, and
success metrics per iteration while giving you a structured approval/briefing
surface for human review.
