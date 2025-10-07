# Testing Cheat Sheet

## Install Dependencies
Run once per virtual environment:

```bash
pip install -r requirements.txt
```

or:

```bash
sh scripts/setup_env.sh
```

## Core Commands
- Quick run with uv (fast startup):
  ```bash
  uv run -m pytest -q
  ```
- Plain Python fallback:
  ```bash
  pytest -q
  ```
- Type checking:
  ```bash
  sh scripts/typecheck
  ```
- Lint + unit test combo (CI parity):
  ```bash
  sh scripts/test
  ```
- End-to-end sanity check:
  ```bash
  sh scripts/smoke_run.sh
  ```

## Targeted Suites
- Dashboard routes: `uv run -m pytest tests/test_dashboard_routes.py`
- Data loading: `uv run -m pytest tests/test_data_loading.py`
- Config defaults: `uv run -m pytest tests/test_config_defaults.py`

## Tips
- Tests rely on relative paths; run them from the repo root.
- Use `AE_PIPELINE_DIR` to point tests at an isolated runs directory when debugging artefact handling.
- When adding new endpoints or artefacts, mirror them with fixtures and assertions in `tests/` so CI catches regressions.
