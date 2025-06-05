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

## Data handling

Aligned OHLCV data is loaded from a directory of CSV files. The
`full_overlap` strategy is recommended as it keeps the maximum number of
datapoints shared across all symbols.  After alignment you can obtain
train/validation/test splits via `evolution_components.get_data_splits`.

## License

This project is licensed under the [MIT License](LICENSE).

