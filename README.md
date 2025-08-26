# Alpha Evolve

[![Tests](https://github.com/Likio3000/Alpha_Evolve/actions/workflows/python.yml/badge.svg)](https://github.com/Likio3000/Alpha_Evolve/actions/workflows/python.yml)

Alpha Evolve is an experiment in evolving alpha factors for systematic trading.

## Requirements

- Python 3.12 or higher
- See `requirements.txt` for Python package dependencies
- Node.js for the optional [pipeline UI](alpha-evolve-pipeline-UI/README.md) which helps configure pipeline parameters

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

You can override the location with the `--data-dir` option when running the
pipeline.

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

## Pre-commit hooks (optional)

To enable fast linting/formatting locally:

```bash
pip install pre-commit
pre-commit install
```

This installs hooks for Ruff (lint + format) and Black.
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

## Limitations and Future Work

The current implementation handles datasets without volume information. While the pipeline runs end-to-end, both the parameters and helper code are tuned to the author's private dataset (e.g. a numeric sector column) and fall short of the low alpha correlations reported in the paper. Increasing the quantity and diversity of data is a focus for future iterations.

The dataset itself cannot be shared due to provider restrictions. In limited experiments the evolved alphas outperformed baseline methods such as the RankLSTM and GA tree, but the results remain below the paper's benchmarks. This project has been an enjoyable learning experience and may be revisited with improved tools and data.

## Further Reading

`Alpha_evolve_paper.pdf` is a reproduction guide describing the original experiment and how to replicate it with this code (kinda).

## License

This project is licensed under the [MIT License](LICENSE).
