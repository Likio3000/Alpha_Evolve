To run tests, first install the project's dependencies:

```bash
pip install -r requirements.txt
```

Then execute the test suite. Using `uv` gives fast startup, but plain
`pytest` works as well:

```bash
uv run -m pytest -q
```

```bash
pytest
```

