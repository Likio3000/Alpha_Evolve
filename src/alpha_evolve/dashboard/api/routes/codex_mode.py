from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.http import HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from ..helpers import ROOT
from ..http import json_error, json_response


CODEX_DIR = Path(os.environ.get("AE_CODEX_DIR", ROOT / "logs" / "codex_mode"))
STATE_FILE = CODEX_DIR / "state.json"
SETTINGS_FILE = CODEX_DIR / "settings.json"
EVENTS_FILE = CODEX_DIR / "run_events.jsonl"
INBOX_FILE = CODEX_DIR / "codex_inbox.md"
EXPERIMENT_LOG = CODEX_DIR / "experiments.md"
REVIEW_FLAG = CODEX_DIR / "review_needed.md"
PID_FILE = CODEX_DIR / "codex_watch.pid"
LOG_FILE = CODEX_DIR / "codex_watch.log"
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


def _ensure_dir() -> None:
    CODEX_DIR.mkdir(parents=True, exist_ok=True)


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
    return state if isinstance(state, dict) else {}


def _read_text(path: Path, limit_lines: Optional[int] = None) -> Optional[str]:
    try:
        if not path.exists():
            return None
        if limit_lines is None:
            return path.read_text(encoding="utf-8", errors="ignore")
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = deque(fh, maxlen=limit_lines)
        return "".join(lines)
    except Exception:
        return None


def _read_events(limit: int = 50) -> List[Dict[str, Any]]:
    if not EVENTS_FILE.exists():
        return []
    try:
        with EVENTS_FILE.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = deque(fh, maxlen=limit)
    except Exception:
        return []
    events: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                events.append(payload)
        except Exception:
            continue
    return events


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _watcher_status() -> Dict[str, Any]:
    pid = None
    running = False
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            running = _pid_running(pid)
        except Exception:
            pid = None
            running = False
    return {"running": running, "pid": pid, "log_file": str(LOG_FILE)}


def summary(_request: HttpRequest):
    _ensure_dir()
    settings = _load_settings()
    state = _load_state()
    events = _read_events(limit=50)
    inbox = _read_text(INBOX_FILE)
    experiments = _read_text(EXPERIMENT_LOG, limit_lines=120)
    review_needed = REVIEW_FLAG.exists()
    session_prompt = _read_text(SESSION_PROMPT)
    status = _watcher_status()
    updated_at = None
    if STATE_FILE.exists():
        try:
            updated_at = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(STATE_FILE.stat().st_mtime)
            )
        except Exception:
            updated_at = None
    return json_response(
        {
            "settings": settings,
            "state": state,
            "events": events,
            "inbox": inbox,
            "experiments": experiments,
            "session_prompt": session_prompt,
            "review_needed": review_needed,
            "watcher": status,
            "updated_at": updated_at,
        }
    )


@csrf_exempt
def update_settings(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    _ensure_dir()
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return json_error("Invalid JSON body", 400)
    if not isinstance(payload, dict):
        return json_error("Settings payload must be a JSON object", 400)
    settings = _load_settings()
    for key in (
        "notify",
        "review_interval",
        "sleep_seconds",
        "yolo_mode",
        "auto_run",
        "auto_run_command",
        "auto_run_mode",
        "auto_run_cooldown",
    ):
        if key in payload:
            settings[key] = payload[key]
    _save_json(SETTINGS_FILE, settings)
    return json_response({"ok": True, "settings": settings})


@csrf_exempt
def start_watcher(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    _ensure_dir()
    status = _watcher_status()
    if status.get("running"):
        return json_response({"started": False, "detail": "Watcher already running."})

    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "codex_watch.py")]
    env = os.environ.copy()
    python_paths = [str(ROOT / "src"), str(ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_FILE.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    finally:
        log_handle.close()
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    return json_response({"started": True, "pid": proc.pid})


@csrf_exempt
def stop_watcher(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    if not PID_FILE.exists():
        return json_response({"stopped": False, "detail": "Watcher not running."})
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        return json_response({"stopped": False, "detail": "Invalid PID file."})
    try:
        os.kill(pid, signal.SIGTERM)
        PID_FILE.unlink(missing_ok=True)
        return json_response({"stopped": True})
    except Exception as exc:
        return json_response({"stopped": False, "detail": str(exc)})
