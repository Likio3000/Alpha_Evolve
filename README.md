# Alpha Evolve

[![Tests](https://github.com/YOUR_GITHUB_USERNAME/Alpha_Evolve/actions/workflows/python.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/Alpha_Evolve/actions/workflows/python.yml)

Alpha Evolve is an experiment in evolving alpha factors for systematic trading.

## Requirements

- Python 3.12 or higher
- See `requirements.txt` for Python package dependencies

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

## Running tests

Before running tests ensure the Python dependencies are installed.  You can do
this directly or via the helper script:

```bash
pip install -r requirements.txt  # or sh scripts/setup_env.sh
```

Once the environment is ready, run pytest.  The `-q` flag gives a concise
summary and should complete without failures:

```bash
pytest -q
```

## Usage

Run the full pipeline for five generations:

```bash
uv run run_pipeline.py 5 --max_lookback_data_option full_overlap --fee 0.5 --debug_prints

Use `--run_baselines` to additionally train the RankLSTM and GA tree baselines.
```

For a longer run using the parameters described in the paper you can simply run

```bash
sh scripts/recommended_pipeline.sh
```

The script expands the meta‑hyper‑parameters and adds a few useful flags like
`--run_baselines`.

For a fully specified command where every parameter is explicitly set you can
run

```bash
sh scripts/run_pipeline_all_args.sh
```

This convenience script mirrors all defaults from `config.py` so you can easily
edit any knob.

The `--debug_prints` flag forwards verbose output to the back-tester.

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
* Sharpe proxy weight **0.0** (combines with mean IC when scoring)
* correlation cutoff for Hall of Fame entries **15 %**
* annualization factor **365 * 6** (default for 4-hour crypto bars)

## Data handling

Aligned OHLC data is loaded from a directory of CSV files. (I did not have access to the volume data) The
`full_overlap` strategy is recommended as it keeps the maximum number of
datapoints shared across all symbols.  After alignment you can obtain
train/validation/test splits via `evolution_components.get_data_splits`.

## License

This project is licensed under the [MIT License](LICENSE).

