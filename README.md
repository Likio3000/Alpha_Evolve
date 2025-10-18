# Alpha Evolve

[![Tests](https://github.com/Likio3000/Alpha_Evolve/actions/workflows/python.yml/badge.svg)](https://github.com/Likio3000/Alpha_Evolve/actions/workflows/python.yml)

Alpha Evolve is an experiment in evolving alpha factors for systematic trading.

> ⚠️ **CLI entrypoints removed:** As of this version the project is driven entirely
> via the dashboard UI/REST API. Legacy commands such as
> `run_pipeline.py …` or `alpha-evolve-pipeline` are no longer available
> as executable scripts. Invoke the pipeline module directly with
> `python -m alpha_evolve.cli.pipeline` or integrate via the dashboard.

## Requirements

- Python 3.12 or higher
- See `requirements.txt` for Python package dependencies
- Optional: Node.js for the advanced pipeline UI (not required). A minimal static dashboard ships in `dashboard-ui/dist` and is served by the Python backend.

## Setup

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively run the helper script:

```bash
sh scripts/setup_env.sh
```

If you change dependencies using `uv add` or by editing `pyproject.toml`,
update `requirements.txt` with:

```bash
sh scripts/update_requirements.sh
```

Many examples use `uv`, a fast Python runtime and package manager. Install it
from <https://github.com/astral-sh/uv> (e.g., `curl -Ls https://astral.sh/uv/install.sh | sh`).
If `uv` is unavailable you can replace `uv` with `python` in the commands.

## Dashboard UI (React + TypeScript)

The dashboard front-end lives in `dashboard-ui/` and is now written in React + TypeScript.

- Source lives under `dashboard-ui/src`; we use Vite for local development and builds.
- Install dependencies with `npm install` (or `pnpm install`) and start a dev server via `npm run dev`.
- Run `npm run build` to regenerate the production bundle in `dashboard-ui/dist/` before deploying.
- The repository ships with a pre-built `dist/` that loads React from a CDN so the Django server keeps working even without Node.js tooling.
- When hacking on the UI in an offline environment, remember to rebuild the bundle once dependencies are available.
- The dashboard header now exposes Backtest Analysis (runs plus charts), Pipeline Controls (launch + activity), and Settings tabs that surface config defaults/presets and let you save curated TOML snapshots.

### Automation Helpers

- `python3 scripts/dev/server_manager.py start` manages the Django/Uvicorn backend with `stop`, `restart`, `status`, and `tail` subcommands. Use `--watch` for auto-restart on backend changes.
- `npm run update:artifacts` wraps `python3 scripts/dev/run_iteration.py`; it runs automatically before `npm run dev` so `artifacts/now_ui` always reflects your latest UI build. Export `AE_SKIP_AUTOCAPTURE=1` to disable every automated capture in that shell.
- `npm run dev` keeps the screenshots fresh during hot reload: after the dev server’s first compile and after each HMR cycle, a lightweight capture runs with `--skip-build --reuse-server --dashboard-url http://127.0.0.1:5173/`, rotating `now_ui → past_ui → past_ui2`.
- `npm run capture:screens` captures Backtest Analysis, Pipeline Controls, and Settings via Playwright, writing `backtest-analysis.png`, `pipeline-controls.png`, and `settings-presets.png` into the active slot.
- `python3 scripts/dev/run_iteration.py` is the orchestration entrypoint: it rotates the three slots, optionally starts the backend, rebuilds the UI, captures screenshots, and copies the backend log into the fresh slot as `dashboard-server.log`.
- See `docs/dev-automation-roadmap.md` for the full automation plan and `docs/iteration-log.md` for the slot layout.

The dev-loop captures assume the Python dashboard backend is already running (e.g. via `python3 scripts/dev/server_manager.py start --watch`); otherwise the Playwright pass will warn about missing data.

## Data Setup

Place your historical OHLC data in the `data` directory at the project root.
Each CSV must contain the columns `time`, `open`, `high`, `low` and `close`. The
`time` column should be Unix epoch seconds.  Store one symbol per file, for
example:

```text
data/
└── AAPL.csv
```

You can override the location with the `--data_dir` option when running the
pipeline.

### Sector mapping (optional)

By default the evolution pipeline uses an empty sector map so every ticker is
assigned `-1` (unknown). This keeps the SP500 workflow simple: the
`sector_id_vector` feature collapses to a constant and sector neutralisation is
equivalent to de‑meaning across the whole universe. If you maintain your own
industry classification, provide it via `sector_mapping` in a custom config or
feed it through `--sector_mapping` JSON overrides.

### Pipeline output location

Run artefacts default to `pipeline_runs_cs/` under the project root. Override the
destination (e.g. a larger external volume) with either the `--output-dir`
flag or the `AE_PIPELINE_DIR` environment variable:

```bash
# Programmatic example
uv run python -m alpha_evolve.cli.pipeline 10 --config configs/sp500.toml --output-dir ~/.alpha-evolve/runs

# Match the dashboard server with the same directory
AE_PIPELINE_DIR=~/.alpha-evolve/runs uv run scripts/run_dashboard.py
```

`run_pipeline.py` updates `pipeline_runs_cs/LATEST` (inside the chosen output
directory) with a project-relative path whenever possible, keeping the Python
dashboard compatible with older runs.

## Running tests

Before running `pytest` you **must** install the project's dependencies:

```bash
pip install -r requirements.txt
```

If you prefer, you can run the helper script instead:

```bash
sh scripts/setup_env.sh
```

Once the environment is ready, prefer running tests with `uv` for fast startup:

```bash
uv run -m pytest -q
```

Alternatively, plain `pytest` also works:

```bash
pytest -q
```

Tip: see `docs/howtos/testing.md` and the helper `scripts/test`.

For a quick end-to-end sanity check, run:

```bash
sh scripts/smoke_run.sh
```

## Guided Self-Evolution

- Run the autonomous controller with `uv run scripts/self_evolve.py` and a search-space file (sample: `configs/self_evolution/sample_equity_space.json`).
- After each pipeline run the agent writes its analysis to `agent_briefings.jsonl` and pauses until you edit `pending_action.json` to approve/reject the next iteration.
- The handshake file also embeds the proposed config; tweak it before setting `status: approved` if you want to adjust parameters manually.
- Artefacts (run dirs, briefing log, pending action, generated configs) live in `pipeline_runs_cs/self_evolution/self_evo_session_*`. The CLI exposes `--auto-approve` for unattended loops.
- Open the dashboard’s Self-Play page (via the top-bar Self-Play button) to see the same information, launch sessions, and send approve/request/stop actions without editing JSON manually. Notes entered there are persisted to `pending_action.json`.
- See `docs/guides/self-evolution.md` for full details on the approval workflow and available configuration knobs.

## Configuration layering (TOML/YAML, env, CLI)

You can now provide configuration via a file and environment variables, with the
following precedence: file < env < CLI.

- Use `--config configs/sp500.toml` (TOML preferred). YAML is supported if
  `PyYAML` is installed.
- Environment variables override file values. Use uppercase keys with prefixes:
  `AE_` for common, `AE_EVO_` (evolution only) and `AE_BT_` (backtest only).
  Examples: `AE_DATA_DIR=./data`, `AE_BT_TOP_TO_BACKTEST=5`.
- CLI flags still take priority over both.

Boolean flags accept a convenient `--no-<flag>` negation when their default is
`True` in the dataclass. For example, to disable sector neutralization during
evolution: `--no-sector_neutralize`.

Console entry points previously installed via `pip`/`uv` have been removed in
favour of the UI workflows.

### Alignment cache controls

To speed up repeated runs, aligned data is cached under `.cache/align_cache` by
default. You can control the cache via CLI or environment variables:

- Disable cache: `--disable-align-cache` or `AE_DISABLE_ALIGN_CACHE=1`
- Custom cache dir: `--align-cache-dir <path>` or `AE_ALIGN_CACHE_DIR=<path>`

### New evolution flags (selection and novelty)

- `--selection_metric`: `ramped` (default), `fixed`, `ic`, `auto`, `phased`.
  - `auto`: ramped early, switches to fixed after the ramp period completes.
  - `phased`: early gens use IC, mid gens use ramped fitness, late gens use fixed fitness. Control with `--ic_phase_gens`.
- `--novelty_boost_w`: diversity bonus for low correlation vs HOF predictions (0 disables).
- `--novelty_struct_w`: structural novelty bonus based on opcode-set Jaccard distance vs HOF (0 disables).
- `--rank_softmax_beta_floor`, `--rank_softmax_beta_target`: soften→sharpen tournament selection weights over the ramp.
- `--hof_corr_mode`: HOF correlation penalty mode: `flat` (default; flattened time × cross-section) or `per_bar` (average of per-bar Spearman correlations).
- `--split_weighting`: combine train/val metrics as `equal` (default) or `by_points`.
- `--ic_tstat_w`: add IC t-stat to fitness to balance magnitude and stability (0 disables).
- `--temporal_decay_half_life`: exponential half-life (in bars) to weight recent bars more in IC/turnover (0 disables).
- Multi-objective selection (Pareto/NSGA-II-lite): `--moea_enabled` to enable, `--moea_elite_frac` to control Pareto-front elites.
- Multi-fidelity evaluation: `--mf_enabled` to turn on a cheap first pass over a truncated window then re-evaluate top candidates fully. Control with `--mf_initial_fraction`, `--mf_promote_fraction`, `--mf_min_promote`.
 - Purged CV across time folds: `--cv_k_folds <K>` and `--cv_embargo <bars>` to use CPCV-like validation when computing selection metrics.

Recommended settings

- `--selection_metric auto` (or `phased` with `--ic_phase_gens 5`)
- `--ramp_fraction 0.33 --ramp_min_gens 5`
- `--novelty_boost_w` in the `0.02–0.05` range when you need extra diversity
- Keep `annualization_factor = 252` for daily SP500 data (already baked into `configs/sp500.toml`)

### Flags overview (quick reference)

- `--config <file>`: Load TOML/YAML; precedence is file < env < CLI.
- `--data_dir <dir>`: Directory of CSVs (`time,open,high,low,close`).
- `--max_lookback_data_option`: `common_1200` | `specific_long_10k` | `full_overlap`.
- `--min_common_points <N>`: Required common eval points (strategy dependent).
- `--eval_lag <N>`: Forward-return lag (1 recommended; required for stops).
- `--workers <N>`: Evolution evaluation workers; try 1–2 if overhead is high.
- `--selection_metric`: `ramped` | `fixed` | `ic` | `auto` | `phased`.
- `--ic_phase_gens <N>`: Length of the IC‑only phase when using `phased`.
- `--novelty_boost_w <w>`: Diversity boost vs HOF (0 disables).
- Cache: `--disable-align-cache`, `--align-cache-dir <path>`.
- Backtest only: `--top_to_backtest`, `--fee`, `--hold`, `--long_short_n`,
  `--stop_loss_pct`, `--annualization_factor`, `--sector_neutralize_positions`,
  `--winsor_p`, `--ensemble_mode`, `--ensemble_size`, `--ensemble_max_corr`.

Evolution-only notable flags: `--novelty_boost_w`, `--novelty_struct_w`,
`--rank_softmax_beta_floor`, `--rank_softmax_beta_target`, `--hof_corr_mode`,
`--split_weighting`, `--ic_tstat_w`, `--temporal_decay_half_life`.

### Tuning guide (practical)

- Exploration → exploitation:
  - Start with `--selection_metric auto` (or `phased` + `--ic_phase_gens 5`).
  - Use `--ramp_fraction 0.33 --ramp_min_gens 5` to avoid “best=gen1”.
- Diversity:
  - Set `--novelty_boost_w 0.02`–`0.05` to prefer novel candidates.
  - Keep `hof_per_gen >= 3`; avoid `keep_dupes_in_hof` unless benchmarking.
- Penalties:
  - If overly harsh early on, lower `corr_penalty_w` or increase ramp length.
  - If churn dominates, increase `turnover_penalty_w` slightly.
- Parallelism:
  - Try `--workers 1` or 2 first. If CPU‑bound and stable, scale up.

## Pre-commit hooks (optional)

To enable fast linting/formatting locally:

```bash
pip install pre-commit
pre-commit install
```

## Usage

Run the full pipeline for five generations:

```bash
# Programmatic example (dashboard UI recommended for day-to-day use)
uv run python -m alpha_evolve.cli.pipeline 5 --max_lookback_data_option full_overlap --fee 0.5 --debug_prints
```

Use `--run_baselines` to additionally train the RankLSTM and GA tree baselines.
Baseline metrics are cached next to the data and reused on subsequent runs.
Pass `--retrain_baselines` to force a fresh training.

For automation you can still import and call
`alpha_evolve.cli.pipeline.run_pipeline_programmatic(...)`; day-to-day usage should happen
through the dashboard UI or the `/api/pipeline/run` endpoint. Any parameter not
explicitly supplied falls back to the defaults defined in
[`alpha_evolve.config`](src/alpha_evolve/config/model.py).

For a longer run using the parameters described in the paper, create a TOML
config (see `configs/sp500.toml`) and run:

```bash
uv run python -m alpha_evolve.cli.pipeline 100 --config configs/sp500.toml
```

CLI flags override config and environment variables (precedence: file < env < CLI).

The `--debug_prints` flag forwards verbose output to the back-tester.
Use `--long_short_n` to trade only the top/bottom N ranked symbols in each
period.  Set it to `0` (default) to use all available symbols.

Logging can be controlled globally via `--log-level` and `--log-file`.  The
`--quiet` and `--debug_prints` flags lower or raise the log verbosity when
running the pipeline.

Back-test summaries now include an `Ops` column showing the operation count of each alpha.

### Selection metric “auto” (explore → exploit)

During evolution you can let the selection switch from ramped fitness to
fixed‑weight fitness automatically after the ramp period:

```bash
# Legacy CLI example (use the dashboard UI instead)
uv run python -m alpha_evolve.cli.pipeline 50 --selection_metric auto --ramp_fraction 0.33 --ramp_min_gens 5
```
This keeps correlation/variance penalties light early (exploration) and then
compares candidates on fixed weights (exploitation) to avoid the “best gen is
first gen” effect you may see with strong, always‑on penalties.

For a tiny end‑to‑end check, use `scripts/smoke_run.sh`.

### Minimal Dashboard UI

Start the iterative API server and open the built-in UI:

```bash
uv run scripts/run_dashboard.py
# then open http://127.0.0.1:8000/ui
```

The UI can:
- Start pipeline runs (dataset + generations + optional overrides)
- Stream live logs and progress (SSE)
- List recent runs and render simple evolution charts from `diagnostics.json`
- Show the backtest summary and per-alpha timeseries charts

Need the richer front-end? See `docs/guides/dashboard-ui.md` for rebuilding the
advanced UI bundle and hooking it up to the FastAPI backend.

For a full end-to-end tour (data prep → pipeline run → dashboard → cleanup),
check `docs/guides/getting-started.md`.

### Adapting new data

As long as your CSVs have the same schema (`time,open,high,low,close` with `time` as epoch seconds or ISO8601) and one file per symbol, the loaders and alignment will work out of the box:

- Place files under a directory and point `--data_dir` to it (or set in config).
- The shared feature builder adds rolling MAs/volatility, range metrics; forward returns are recomputed after alignment.
- If you don’t have a sector mapping for your universe, the `sector_id_vector` will default to `-1` (single bucket). You can disable sector neutralization with `--no-sector_neutralize` during evolution and `--sector_neutralize_positions false` in backtests.

## Default hyperparameters

The values in `alpha_evolve.config` mirror the meta‑hyper‑parameters listed in
Section 4.1 of the reproduction guide:

* population size **100**
* tournament size **10**
* mutation probability **0.9**
* setup, predict and update operation limits **21/21/45**
* scalar/vector/matrix operand limits **10/16/4**
* evaluation cache size **128**
* correlation cutoff for Hall of Fame entries **15 %**
* Sharpe proxy weight **0**
* annualization factor **252** (trading days per year for daily data)
* long/short universe size **0** (use all symbols)
* `factor_penalty_w` default **0.0** (set >0 to penalize exposure to style factors listed in `factor_penalty_factors`, e.g. `ret1d_t,vol20_t,range_rel_t`)
* `evaluation_horizons` default **(1,)** (supply additional horizons such as `(1, 3, 6)` to score alphas on multiple holding periods; metrics are averaged across horizons)
* `qd_archive_enabled` default **false** – enable a MAP-Elites style archive that keeps turnover/complexity-diverse elites (`qd_turnover_bins`, `qd_complexity_bins`, `qd_max_entries`)

## Data handling

Aligned OHLC data is loaded from a directory of CSV files. (I did not have access to the volume data) The
`full_overlap` strategy is recommended as it keeps the maximum number of
datapoints shared across all symbols.  After alignment you can obtain
train/validation/test splits via `alpha_evolve.evolution.data.get_data_splits`.

### Quickstart: SP500

Fetch 20y of split/dividend‑adjusted OHLC from Yahoo and run a short pipeline:

```bash
python scripts/fetch_sp500_data.py --out data_sp500 --years 20
# Programmatic example (dashboard UI recommended for day-to-day use)
uv run python -m alpha_evolve.cli.pipeline 5 --config configs/sp500.toml --selection_metric auto \
  --disable-align-cache --debug_prints
```
By default SP500 runs don’t use a custom sector mapping; you can disable sector
neutralization explicitly if desired.

## Limitations and Future Work

The current implementation handles datasets without volume information. While the pipeline runs end-to-end, both the parameters and helper code are tuned to the author's private dataset (e.g. a numeric sector column) and fall short of the low alpha correlations reported in the paper. Increasing the quantity and diversity of data is a focus for future iterations.

The dataset itself cannot be shared due to provider restrictions. In limited experiments the evolved alphas outperformed baseline methods such as the RankLSTM and GA tree, but the results remain below the paper's benchmarks. This project has been an enjoyable learning experience and may be revisited with improved tools and data.

## Further Reading

`Alpha_evolve_paper.pdf` is a reproduction guide describing the original experiment and how to replicate it with this code (kinda).

## Utilities

- `sh scripts/typecheck` – mypy quick pass (uses `uv run` when available).
- `python scripts/cleanup_runs.py --help` – prune or inspect pipeline runs.

## License

This project is licensed under the [MIT License](LICENSE).
