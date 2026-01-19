#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("AE_OUTPUT_DIR", ROOT / "pipeline_runs_cs"))
LOG_DIR = OUTPUT_DIR / "logs"
CODEX_DIR = Path(os.environ.get("AE_CODEX_DIR", LOG_DIR / "codex_mode"))

STATE_FILE = CODEX_DIR / "state.json"
SETTINGS_FILE = CODEX_DIR / "settings.json"
EVENTS_FILE = CODEX_DIR / "run_events.jsonl"
INBOX_FILE = CODEX_DIR / "codex_inbox.md"
EXPERIMENT_LOG = CODEX_DIR / "experiments.md"
REVIEW_FLAG = CODEX_DIR / "review_needed.md"
AUTOPROMPT_FILE = CODEX_DIR / "autoprompt.md"
AUTORUN_LOCK = CODEX_DIR / "codex_autorun.lock"
SESSION_PROMPT = ROOT / "docs" / "codex_mode" / "SESSION_PROMPT.md"

DEFAULT_SETTINGS = {
    "notify": True,
    "review_interval": 3,
    "sleep_seconds": 15,
    "yolo_mode": False,
    "auto_run": False,
    "auto_run_command": "codex",
    "auto_run_mode": "terminal",
    "auto_run_cooldown": 300,
}

MAX_SEEN = 500


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _save_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
    tmp.replace(path)


def _ensure_dir() -> None:
    CODEX_DIR.mkdir(parents=True, exist_ok=True)


def _load_settings() -> Dict[str, Any]:
    settings = _load_json(SETTINGS_FILE, {})
    if not isinstance(settings, dict):
        settings = {}
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings)
    _save_json(SETTINGS_FILE, merged)
    return merged


def _load_state() -> Dict[str, Any]:
    state = _load_json(STATE_FILE, {})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("seen_pipeline_runs", [])
    state.setdefault("seen_ml_runs", [])
    state.setdefault("runs_since_review", 0)
    return state


def _bootstrap_state(state: Dict[str, Any]) -> None:
    seen_pipeline = []
    for run_dir in OUTPUT_DIR.glob("run_*"):
        bt_dir = run_dir / "backtest_portfolio_csvs"
        if list(bt_dir.glob("backtest_summary_top*.csv")):
            seen_pipeline.append(run_dir.name)
    state["seen_pipeline_runs"] = seen_pipeline[-MAX_SEEN:]

    seen_ml = []
    ml_root = OUTPUT_DIR / "ml_runs"
    if ml_root.exists():
        for run_dir in ml_root.glob("run_*"):
            if (run_dir / "ml_summary.json").exists():
                seen_ml.append(run_dir.name)
    state["seen_ml_runs"] = seen_ml[-MAX_SEEN:]

def _save_state(state: Dict[str, Any]) -> None:
    _save_json(STATE_FILE, state)


def _sanitize_message(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _notify(title: str, message: str, enabled: bool) -> None:
    if not enabled:
        return
    if sys.platform == "darwin":
        script = (
            f'display notification "{_sanitize_message(message)}" '
            f'with title "{_sanitize_message(title)}"'
        )
        subprocess.run(["osascript", "-e", script], check=False)
    else:
        print(f"[notify] {title}: {message}")


def _append_event(event: Dict[str, Any]) -> None:
    line = json.dumps(event, ensure_ascii=True)
    with EVENTS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _ensure_experiment_log() -> None:
    if EXPERIMENT_LOG.exists():
        return
    template = (
        "# Codex Mode Experiment Log\n\n"
        "Use this log to track ideas, changes, and results.\n\n"
        "## Active Focus\n\n"
        "- Objective:\n"
        "- Current hypothesis:\n"
        "- Next run plan:\n\n"
        "## Run Notes\n\n"
        "- YYYY-MM-DD HH:MM:SS | <run_type> | sharpe=<value> | notes=\n\n"
    )
    EXPERIMENT_LOG.write_text(template, encoding="utf-8")


def _update_inbox(message: str) -> None:
    payload = (
        "# Codex Inbox\n\n"
        f"Last update: {_now_ts()}\n\n"
        "## Latest Event\n\n"
        f"{message}\n\n"
        "## Immediate Actions\n\n"
        "- Inspect the run outputs and metrics.\n"
        "- Decide the next run or code change to improve Sharpe.\n"
        "- Update the experiment log with findings.\n\n"
    )
    INBOX_FILE.write_text(payload, encoding="utf-8")


def _best_sharpe_from_csv(path: Path) -> Optional[float]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            best = None
            for row in reader:
                try:
                    value = float(row.get("Sharpe", "nan"))
                except Exception:
                    continue
                if best is None or value > best:
                    best = value
            return best
    except Exception:
        return None


def _scan_pipeline_runs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen_list = list(state.get("seen_pipeline_runs", []))
    seen = set(seen_list)
    runs = sorted(OUTPUT_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime)
    for run_dir in runs:
        name = run_dir.name
        if name in seen:
            continue
        bt_dir = run_dir / "backtest_portfolio_csvs"
        summary_files = sorted(bt_dir.glob("backtest_summary_top*.csv"))
        if not summary_files:
            continue
        sharpe = _best_sharpe_from_csv(summary_files[-1])
        event = {
            "ts": _now_ts(),
            "kind": "pipeline",
            "run_name": name,
            "run_dir": str(run_dir),
            "sharpe": sharpe,
        }
        events.append(event)
        seen.add(name)
        seen_list.append(name)
    if len(seen_list) > MAX_SEEN:
        seen_list = seen_list[-MAX_SEEN:]
    state["seen_pipeline_runs"] = seen_list
    return events


def _scan_ml_runs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen_list = list(state.get("seen_ml_runs", []))
    seen = set(seen_list)
    ml_root = OUTPUT_DIR / "ml_runs"
    if not ml_root.exists():
        return events
    runs = sorted(ml_root.glob("run_*"), key=lambda p: p.stat().st_mtime)
    for run_dir in runs:
        name = run_dir.name
        if name in seen:
            continue
        summary_path = run_dir / "ml_summary.json"
        meta_path = run_dir / "meta" / "run_metadata.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path, {})
        meta = _load_json(meta_path, {})
        status = meta.get("status")
        if status not in ("complete", "error"):
            completed = summary.get("completed")
            total = summary.get("total")
            if isinstance(completed, int) and isinstance(total, int) and total > 0:
                if completed < total:
                    continue
            else:
                continue
        sharpe = summary.get("best_sharpe")
        event = {
            "ts": _now_ts(),
            "kind": "ml",
            "run_name": name,
            "run_dir": str(run_dir),
            "status": status,
            "sharpe": sharpe,
        }
        events.append(event)
        seen.add(name)
        seen_list.append(name)
    if len(seen_list) > MAX_SEEN:
        seen_list = seen_list[-MAX_SEEN:]
    state["seen_ml_runs"] = seen_list
    return events


def _append_experiment_line(event: Dict[str, Any]) -> None:
    line = (
        f"- {event.get('ts')} | {event.get('kind')} | "
        f"sharpe={event.get('sharpe')} | run={event.get('run_name') or event.get('label')}\n"
    )
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _mark_review_needed() -> None:
    REVIEW_FLAG.write_text(
        "# Review Needed\n\n"
        "A review cadence trigger fired. Inspect uncommitted changes, "
        "fix issues, and update the experiment log.\n",
        encoding="utf-8",
    )


def _read_text(path: Path, limit_lines: Optional[int] = None) -> str:
    try:
        if not path.exists():
            return ""
        if limit_lines is None:
            return path.read_text(encoding="utf-8", errors="ignore")
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = deque(fh, maxlen=limit_lines)
        return "".join(lines)
    except Exception:
        return ""


def _escape_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_autoprompt(event: Dict[str, Any]) -> Path:
    prompt = _read_text(SESSION_PROMPT)
    inbox = _read_text(INBOX_FILE)
    experiments = _read_text(EXPERIMENT_LOG, limit_lines=80)
    payload = (
        f"{prompt.strip()}\n\n"
        "## Latest Run Event\n\n"
        f"{json.dumps(event, indent=2)}\n\n"
        "## Codex Inbox\n\n"
        f"{inbox.strip()}\n\n"
        "## Experiment Log (tail)\n\n"
        f"{experiments.strip()}\n"
    )
    AUTOPROMPT_FILE.write_text(payload, encoding="utf-8")
    return AUTOPROMPT_FILE


def _lock_active() -> bool:
    if not AUTORUN_LOCK.exists():
        return False
    try:
        data = _load_json(AUTORUN_LOCK, {})
        if not isinstance(data, dict):
            return False
        expires = data.get("expires_at")
        if isinstance(expires, (int, float)) and time.time() < float(expires):
            return True
    except Exception:
        return False
    return False


def _set_lock(cooldown: int) -> None:
    payload = {"created_at": time.time(), "expires_at": time.time() + cooldown}
    _save_json(AUTORUN_LOCK, payload)


def _launch_codex(prompt_path: Path, settings: Dict[str, Any]) -> None:
    command = str(settings.get("auto_run_command") or "codex").strip()
    if not command:
        return
    mode = str(settings.get("auto_run_mode") or "terminal").strip().lower()
    if "{prompt_file}" in command:
        cmd = command.replace("{prompt_file}", str(prompt_path))
    else:
        cmd = f"{command} < {prompt_path}"

    if mode == "terminal" and sys.platform == "darwin":
        script = f'tell application "Terminal" to do script "cd {ROOT} && {cmd}"'
        subprocess.run(["osascript", "-e", _escape_applescript(script)], check=False)
        return

    subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=str(ROOT),
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _process_events(events: List[Dict[str, Any]], settings: Dict[str, Any], state: Dict[str, Any]) -> None:
    if not events:
        return
    _ensure_experiment_log()
    for event in events:
        _append_event(event)
        _append_experiment_line(event)
        run_label = event.get("run_name") or event.get("label") or "run"
        sharpe = event.get("sharpe")
        msg = f"{event.get('kind')} finished: {run_label} (sharpe={sharpe})"
        _update_inbox(msg)
        _notify("Alpha Evolve", msg, bool(settings.get("notify", True)))
        state["runs_since_review"] = int(state.get("runs_since_review", 0) or 0) + 1
        interval = int(settings.get("review_interval", 3) or 3)
        if interval > 0 and state["runs_since_review"] >= interval:
            state["runs_since_review"] = 0
            _mark_review_needed()
            _notify(
                "Alpha Evolve",
                "Review cadence reached: inspect uncommitted changes.",
                bool(settings.get("notify", True)),
            )
        if bool(settings.get("auto_run", False)):
            cooldown = int(settings.get("auto_run_cooldown", 300) or 300)
            if not _lock_active():
                prompt_path = _build_autoprompt(event)
                _set_lock(cooldown)
                _launch_codex(prompt_path, settings)
                state["last_autorun_ts"] = time.time()


def run_once(settings: Dict[str, Any], state: Dict[str, Any]) -> None:
    events: List[Dict[str, Any]] = []
    events.extend(_scan_pipeline_runs(state))
    events.extend(_scan_ml_runs(state))
    _process_events(events, settings, state)
    state["last_scan"] = time.time()


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch for run completions and notify Codex.")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit.")
    parser.add_argument("--interval", type=int, help="Override sleep interval (seconds).")
    args = parser.parse_args()

    _ensure_dir()
    settings = _load_settings()
    if args.interval:
        settings["sleep_seconds"] = int(args.interval)
        _save_json(SETTINGS_FILE, settings)
    state = _load_state()
    if not STATE_FILE.exists():
        _bootstrap_state(state)
        _save_state(state)
        if args.once:
            return 0

    if args.once:
        run_once(settings, state)
        _save_state(state)
        return 0

    while True:
        settings = _load_settings()
        run_once(settings, state)
        _save_state(state)
        time.sleep(int(settings.get("sleep_seconds", 15)))


if __name__ == "__main__":
    raise SystemExit(main())
