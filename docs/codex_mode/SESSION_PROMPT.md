You are in Codex Mode for Alpha Evolve. The goal is to maximize Sharpe.

First steps:
- Open `logs/codex_mode/codex_inbox.md` and `logs/codex_mode/experiments.md`.
- Check `logs/codex_mode/run_events.jsonl` for the latest results.
- If `logs/codex_mode/review_needed.md` exists, review uncommitted changes.

When a run finishes:
- Inspect the run results and compare Sharpe to recent baselines.
- Decide what to tune (parameters, risk controls, model presets, or code).
- Queue the next run and log the rationale + outcome in `experiments.md`.

If YOLO mode is enabled in `logs/codex_mode/settings.json`, you may change
code, UI, or tests to improve Sharpe. Keep notes and run tests afterward.
