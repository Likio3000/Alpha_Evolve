# Alpha Evolve

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

## Running tests

Use `pytest` to run the unit tests after installing the dependencies:

```bash
pytest
```

## Data handling

Aligned OHLCV data is loaded from a directory of CSV files. The
`full_overlap` strategy is recommended as it keeps the maximum number of
datapoints shared across all symbols.  After alignment you can obtain
train/validation/test splits via `evolution_components.get_data_splits`.
