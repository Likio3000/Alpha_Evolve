from __future__ import annotations

import asyncio
import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..helpers import PIPELINE_DIR, ROOT
from ..jobs import STATE
from ..models import SelfplayApprovalRequest, SelfplayRunRequest


router = APIRouter()


SELFPLAY_ROOT = PIPELINE_DIR / "self_evolution"


def _history_path() -> Path:
    return PIPELINE_DIR / "selfplay_history.json"


def _load_history() -> List[Dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to read self-play history: {exc}")
    raise HTTPException(status_code=500, detail="Invalid self-play history format; expected a list")


def _list_session_dirs() -> List[Path]:
    if not SELFPLAY_ROOT.exists():
        return []
    dirs = [p for p in SELFPLAY_ROOT.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name)
    return dirs


def _resolve_session(session: Optional[str]) -> Optional[Path]:
    if session:
        raw = Path(session)
        if raw.is_absolute():
            try:
                raw.relative_to(SELFPLAY_ROOT)
            except ValueError:
                raise HTTPException(status_code=400, detail="Session path must live under self_evolution directory")
            candidate = raw
        else:
            candidate = (SELFPLAY_ROOT / raw).resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise HTTPException(status_code=404, detail="Requested session not found")
        try:
            candidate.relative_to(SELFPLAY_ROOT)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session path")
        return candidate
    dirs = _list_session_dirs()
    return dirs[-1] if dirs else None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read {path.name}: {exc}")
    return None


def _load_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read {path.name}: {exc}")
    entries: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    entries.reverse()
    return entries


def _relative_to_pipeline(path: Path) -> str:
    try:
        return str(path.relative_to(PIPELINE_DIR))
    except ValueError:
        return str(path)


@router.post("/api/selfplay/run")
async def start_selfplay_run(payload: SelfplayRunRequest) -> Dict[str, Any]:
    search_path = Path(payload.search_space).expanduser()
    if not search_path.is_absolute():
        search_path = (ROOT / search_path).resolve()
    if not search_path.exists():
        raise HTTPException(status_code=404, detail=f"Search-space file not found: {search_path}")

    config_path: Optional[Path] = None
    if payload.config:
        candidate = Path(payload.config).expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        if not candidate.exists():
            raise HTTPException(status_code=404, detail=f"Config not found: {candidate}")
        config_path = candidate

    job_id = str(uuid.uuid4())
    q = STATE.new_queue(job_id)

    args: list[str] = [
        "uv",
        "run",
        str(ROOT / "scripts" / "self_evolve.py"),
        "--search-space",
        str(search_path),
    ]

    if config_path:
        args += ["--config", str(config_path)]
    if payload.iterations is not None:
        args += ["--iterations", str(int(payload.iterations))]
    if payload.seed is not None:
        args += ["--seed", str(int(payload.seed))]
    if payload.objective:
        args += ["--objective", payload.objective]
    if payload.minimize:
        args.append("--minimize")
    if payload.exploration_prob is not None:
        args += ["--exploration-prob", str(float(payload.exploration_prob))]
    if payload.auto_approve:
        args.append("--auto-approve")
    if payload.pipeline_output_dir:
        args += ["--pipeline-output-dir", payload.pipeline_output_dir]
    if payload.session_root:
        args += ["--session-root", payload.session_root]
    if payload.approval_poll_interval is not None:
        args += ["--approval-poll-interval", str(float(payload.approval_poll_interval))]
    if payload.approval_timeout is not None:
        args += ["--approval-timeout", str(float(payload.approval_timeout))]
    if payload.pipeline_log_level:
        args += ["--pipeline-log-level", payload.pipeline_log_level]
    if payload.debug_prints:
        args.append("--debug-prints")
    if payload.run_baselines:
        args.append("--run-baselines")
    if payload.retrain_baselines:
        args.append("--retrain-baselines")

    async def _pump() -> None:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        proc = subprocess.Popen(
            args,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        STATE.set_proc(job_id, proc)
        q.put_nowait(json.dumps({"type": "status", "msg": "started", "args": args}))

        loop = asyncio.get_running_loop()

        def _stream_output() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    try:
                        print(line, flush=True)
                    except Exception:
                        pass
                    loop.call_soon_threadsafe(STATE.add_log, job_id, line)
                    loop.call_soon_threadsafe(q.put_nowait, json.dumps({"type": "log", "raw": line}))
            finally:
                code = proc.wait()

                def _finalize() -> None:
                    q.put_nowait(json.dumps({"type": "status", "msg": "exit", "code": code}))
                    STATE.procs.pop(job_id, None)

                loop.call_soon_threadsafe(_finalize)

        await asyncio.to_thread(_stream_output)

    asyncio.create_task(_pump())
    return {"job_id": job_id}


@router.get("/api/selfplay/history")
def get_selfplay_history(limit: int = Query(default=25, ge=1, le=200)) -> Dict[str, Any]:
    entries = _load_history()
    if not entries:
        return {"history": []}
    entries_sorted = sorted(entries, key=lambda item: (item.get("timestamp"), item.get("iteration", -1)))
    return {"history": entries_sorted[-limit:][::-1]}


@router.get("/api/selfplay/history/latest")
def get_latest_selfplay() -> Dict[str, Any]:
    entries = _load_history()
    if not entries:
        return {"entry": None}
    latest = max(entries, key=lambda item: (item.get("timestamp"), item.get("iteration", -1)))
    return {"entry": latest}


@router.get("/api/selfplay/status")
def get_selfplay_status(
    session: Optional[str] = Query(default=None, description="Session directory (relative or absolute)"),
    limit: int = Query(default=5, ge=1, le=50),
) -> Dict[str, Any]:
    session_dir = _resolve_session(session)
    if session_dir is None:
        return {
            "session_dir": None,
            "session_name": None,
            "pending_action": None,
            "briefings": [],
            "summary": None,
        }

    summary = _load_json(session_dir / "session_summary.json")
    pending = _load_json(session_dir / "pending_action.json")
    briefings = _load_jsonl(session_dir / "agent_briefings.jsonl", limit)

    return {
        "session_dir": str(session_dir),
        "session_name": session_dir.name,
        "session_path": _relative_to_pipeline(session_dir),
        "pending_action": pending,
        "briefings": briefings,
        "summary": summary,
    }


@router.post("/api/selfplay/approval")
def post_selfplay_approval(payload: SelfplayApprovalRequest) -> Dict[str, Any]:
    session_dir = _resolve_session(payload.session)
    if session_dir is None:
        raise HTTPException(status_code=404, detail="No self-play session found")

    pending_path = session_dir / "pending_action.json"
    data = _load_json(pending_path)
    if data is None:
        raise HTTPException(status_code=404, detail="pending_action.json not found for session")

    status = payload.status.strip().lower()
    if not status:
        raise HTTPException(status_code=400, detail="Status must be non-empty")

    data["status"] = status
    if payload.notes is not None:
        data["notes"] = payload.notes
    if payload.approved_candidate is not None:
        data["approved_candidate"] = payload.approved_candidate
    if payload.approved_updates is not None:
        data["approved_updates"] = payload.approved_updates
    data["modified_at"] = datetime.utcnow().isoformat() + "Z"

    try:
        pending_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update pending action: {exc}")

    return {
        "session_dir": str(session_dir),
        "pending_action": data,
    }
