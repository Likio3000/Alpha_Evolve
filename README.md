# Alpha Evolve

[![Tests](https://github.com/Likio3000/Alpha_Evolve/actions/workflows/python.yml/badge.svg)](https://github.com/Likio3000/Alpha_Evolve/actions/workflows/python.yml)

Alpha Evolve is an experiment in evolving alpha factors for systematic trading.

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

## Data Setup

Place your historical OHLC data in the `data` directory at the project root.
Each CSV must contain the columns `time`, `open`, `high`, `low` and `close`. The
`time` column should be Unix epoch seconds.  Store one symbol per file, for
example:

```text
data/
└── BTC.csv
```

You can override the location with the `--data_dir` option when running the
pipeline.

### Sector mapping (crypto vs SP500)

The project ships with a simple crypto‑oriented sector mapping used by some
operators and for optional sector‑neutralization during evolution/backtests. For
symbols not present in the mapping (e.g., most SP500 tickers), we assign `-1`
as a catch‑all sector. In that case:

- The `sector_id_vector` feature is a constant `-1` vector.
- Sector neutralization degenerates to global de‑meaning (equivalent to
  subtracting the cross‑sectional mean), so it’s safe but redundant.

Recommendations:

- Crypto: leave sector neutralization enabled (default).
- SP500: you can disable it for clarity with `--no-sector_neutralize` on
  evolution and `--sector_neutralize_positions false` on backtests, or set those
  in `configs/sp500.toml`.

## Running tests

Before running `pytest` you **must** install the project's dependencies:

```bash
pip install -r requirements.txt
```

If you prefer, you can run the helper script instead:

```bash
sh scripts/setup_env.sh
```

Once the environment is ready, run `pytest`.  The `-q` flag gives a concise
summary and should complete without failures:

```bash
pytest -q
```

For a quick end-to-end sanity check, run:

```bash
sh scripts/smoke_run.sh
```

## Configuration layering (TOML/YAML, env, CLI)

You can now provide configuration via a file and environment variables, with the
following precedence: file < env < CLI.

- Use `--config configs/crypto.toml` (TOML preferred). YAML is supported if
  `PyYAML` is installed.
- Environment variables override file values. Use uppercase keys with prefixes:
  `AE_` for common, `AE_EVO_` (evolution only) and `AE_BT_` (backtest only).
  Examples: `AE_DATA_DIR=./data`, `AE_BT_TOP_TO_BACKTEST=5`.
- CLI flags still take priority over both.

Boolean flags accept a convenient `--no-<flag>` negation when their default is
`True` in the dataclass. For example, to disable sector neutralization during
evolution: `--no-sector_neutralize`.

Console entry points are available when installed:

- `alpha-evolve-pipeline` → `run_pipeline.py`
- `alpha-evolve-backtest` → `backtest_evolved_alphas.py`

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

- Crypto:
  - `--selection_metric auto` (or `phased` with `--ic_phase_gens 5`)
  - `--ramp_fraction 0.33 --ramp_min_gens 5`
  - `--novelty_boost_w 0.02` to `0.05`
  - Keep `sector_neutralize` enabled (default)
- SP500:
  - Same selection settings as above
  - Disable sector neutralization (already disabled in `configs/sp500.toml`)
  - `annualization_factor = 252` (already in `configs/sp500.toml`)

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
uv run run_pipeline.py 5 --max_lookback_data_option full_overlap --fee 0.5 --debug_prints
```

Use `--run_baselines` to additionally train the RankLSTM and GA tree baselines.
Baseline metrics are cached next to the data and reused on subsequent runs.
Pass `--retrain_baselines` to force a fresh training.

Call `run_pipeline.py` directly for quick experiments or when overriding just a
few options.  Any parameter not supplied on the command line falls back to the
defaults defined in [`config.py`](config.py).

For a longer run using the parameters described in the paper, create a TOML
config (see `configs/crypto.toml` and `configs/sp500.toml`) and run:

```bash
alpha-evolve-pipeline 100 --config configs/crypto.toml
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
uv run run_pipeline.py 50 --selection_metric auto --ramp_fraction 0.33 --ramp_min_gens 5
```
This keeps correlation/variance penalties light early (exploration) and then
compares candidates on fixed weights (exploitation) to avoid the “best gen is
first gen” effect you may see with strong, always‑on penalties.

For a tiny end‑to‑end check, use `scripts/smoke_run.sh`.

### Minimal Dashboard UI

Start the iterative API server and open the built‑in UI:

```bash
uv run scripts/iterative_dashboard_server.py
# then open http://127.0.0.1:8000/ui
```

The UI can:
- Start pipeline runs (dataset + generations + optional overrides)
- Stream live logs and progress (SSE)
- List recent runs and render simple evolution charts from `diagnostics.json`
- Show the backtest summary and per‑alpha timeseries charts

### Adapting new data

As long as your CSVs have the same schema (`time,open,high,low,close` with `time` as epoch seconds or ISO8601) and one file per symbol, the loaders and alignment will work out of the box:

- Place files under a directory and point `--data_dir` to it (or set in config).
- The shared feature builder adds rolling MAs/volatility, range metrics; forward returns are recomputed after alignment.
- If you don’t have a sector mapping for your universe, the `sector_id_vector` will default to `-1` (single bucket). You can disable sector neutralization with `--no-sector_neutralize` during evolution and `--sector_neutralize_positions false` in backtests.

## Default hyperparameters

The values in `config.py` mirror the meta‑hyper‑parameters listed in
Section 4.1 of the reproduction guide:

* population size **100**
* tournament size **10**
* mutation probability **0.9**
* setup, predict and update operation limits **21/21/45**
* scalar/vector/matrix operand limits **10/16/4**
* evaluation cache size **128**
* correlation cutoff for Hall of Fame entries **15 %**
* Sharpe proxy weight **0**
* annualization factor **365 * 6** (default for 4-hour crypto bars)
* long/short universe size **0** (use all symbols)

## Data handling

Aligned OHLC data is loaded from a directory of CSV files. (I did not have access to the volume data) The
`full_overlap` strategy is recommended as it keeps the maximum number of
datapoints shared across all symbols.  After alignment you can obtain
train/validation/test splits via `evolution_components.get_data_splits`.

### Quickstart: SP500

Fetch 20y of split/dividend‑adjusted OHLC from Yahoo and run a short pipeline:

```bash
python scripts/fetch_sp500_data.py --out data_sp500 --years 20
uv run run_pipeline.py 5 --config configs/sp500.toml --selection_metric auto \
  --disable-align-cache --debug_prints
```
By default SP500 runs don’t use a custom sector mapping; you can disable sector
neutralization explicitly if desired.

## Limitations and Future Work

The current implementation handles datasets without volume information. While the pipeline runs end-to-end, both the parameters and helper code are tuned to the author's private dataset (e.g. a numeric sector column) and fall short of the low alpha correlations reported in the paper. Increasing the quantity and diversity of data is a focus for future iterations.

The dataset itself cannot be shared due to provider restrictions. In limited experiments the evolved alphas outperformed baseline methods such as the RankLSTM and GA tree, but the results remain below the paper's benchmarks. This project has been an enjoyable learning experience and may be revisited with improved tools and data.

## Further Reading

`Alpha_evolve_paper.pdf` is a reproduction guide describing the original experiment and how to replicate it with this code (kinda).

## License

This project is licensed under the [MIT License](LICENSE).
