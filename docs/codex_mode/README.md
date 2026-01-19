# Codex Mode

Codex Mode is a workflow that keeps the project improving Sharpe continuously.
It combines a run watcher, experiment logging, and a prompt handoff for new
Codex sessions.

## What it does

- Watches for run completions (pipeline, ML lab) and notifies you.
- Logs every completed run with Sharpe into a structured experiment log.
- Triggers a review reminder every N runs to inspect uncommitted changes.
- Provides a session prompt so new Codex sessions know what to do.

## Files created at runtime

The watcher writes to `logs/codex_mode/` (ignored by git):

- `state.json`: last seen runs and counters.
- `settings.json`: notification + review interval settings.
- `run_events.jsonl`: one JSON event per completed run.
- `experiments.md`: running notes and results.
- `codex_inbox.md`: most recent action prompt.
- `review_needed.md`: created every N runs to force a review.

## Start / stop the watcher

Start (background):

```
./scripts/start_codex_mode.sh
```

Stop:

```
./scripts/stop_codex_mode.sh
```

If you want to run a single scan:

```
python scripts/codex_watch.py --once
```

The first run bootstraps state and will not notify for historical runs. To
force notifications for existing runs, delete `logs/codex_mode/state.json`.

## Settings

`logs/codex_mode/settings.json` controls the workflow:

```
{
  "notify": true,
  "review_interval": 3,
  "sleep_seconds": 15,
  "yolo_mode": false,
  "auto_run": false,
  "auto_run_command": "codex",
  "auto_run_mode": "terminal",
  "auto_run_cooldown": 300
}
```

- `review_interval`: set to 3 or 5 based on your preference.
- `yolo_mode`: when true, new Codex sessions are allowed to change code/tests/UI
  aggressively in service of higher Sharpe.
- `auto_run`: when true, starts a Codex session automatically after each run.
- `auto_run_command`: command to start Codex (supports `{prompt_file}` placeholder).
- `auto_run_mode`: `terminal` (default) opens a new Terminal tab on macOS, `background` runs silently.
- `auto_run_cooldown`: seconds before another auto-run can fire.

## Run loops

Recommended loop:

1. Start a pipeline run from the UI (or CLI).
2. Keep the watcher running to collect results.
3. After each run:
   - Inspect metrics and timeseries.
   - Decide next parameter or code change.
   - Log findings in `experiments.md`.
4. Every 3-5 runs, honor the review reminder to clean up changes.

## Session handoff

Use the prompt template in `docs/codex_mode/SESSION_PROMPT.md` for each new
Codex session.
