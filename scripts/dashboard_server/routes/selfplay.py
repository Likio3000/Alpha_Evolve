from __future__ import annotations

import asyncio
import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.http import HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from pydantic import ValidationError

from ..helpers import PIPELINE_DIR, ROOT
from ..http import json_error, json_response
from ..jobs import STATE
from ..models import SelfplayApprovalRequest, SelfplayRunRequest


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
        raise RuntimeError(f"Failed to read self-play history: {exc}")
    raise RuntimeError("Invalid self-play history format; expected a list")


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
                raise ValueError("Session path must live under self_evolution directory")
            candidate = raw
        else:
            candidate = (SELFPLAY_ROOT / raw).resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError("Requested session not found")
        try:
            candidate.relative_to(SELFPLAY_ROOT)
        except ValueError:
            raise ValueError("Invalid session path")
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
        raise RuntimeError(f"Failed to read {path.name}: {exc}")
    return None


def _load_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path.name}: {exc}")
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


@csrf_exempt
async def start_selfplay_run(request: HttpRequest) -> Any:
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return json_error("Invalid JSON body", 400)
    try:
        payload = SelfplayRunRequest.model_validate(data)
    except ValidationError as exc:
        return json_response({"detail": exc.errors()}, status=422)

    search_path = Path(payload.search_space).expanduser()
    if not search_path.is_absolute():
        search_path = (ROOT / search_path).resolve()
    if not search_path.exists():
        return json_error(f"Search-space file not found: {search_path}", 404)

    config_path: Optional[Path] = None
    if payload.config:
        candidate = Path(payload.config).expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        if not candidate.exists():
            return json_error(f"Config not found: {candidate}", 404)
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
    return json_response({"job_id": job_id})


def get_selfplay_history(request: HttpRequest):
    limit_param = request.GET.get("limit", "25")
    try:
        limit = max(1, min(200, int(limit_param)))
    except Exception:
        return json_error("limit must be an integer", 400)
    try:
        entries = _load_history()
    except RuntimeError as exc:
        return json_error(str(exc), 500)
    if not entries:
        return json_response({"history": []})
    entries_sorted = sorted(entries, key=lambda item: (item.get("timestamp"), item.get("iteration", -1)))
    return json_response({"history": entries_sorted[-limit:][::-1]})


def get_latest_selfplay(request: HttpRequest):
    try:
        entries = _load_history()
    except RuntimeError as exc:
        return json_error(str(exc), 500)
    if not entries:
        return json_response({"entry": None})
    latest = max(entries, key=lambda item: (item.get("timestamp"), item.get("iteration", -1)))
    return json_response({"entry": latest})


def get_selfplay_status(request: HttpRequest):
    session = request.GET.get("session")
    limit_param = request.GET.get("limit", "5")
    try:
        limit = max(1, min(50, int(limit_param)))
    except Exception:
        return json_error("limit must be an integer", 400)
    try:
        session_dir = _resolve_session(session)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except FileNotFoundError:
        return json_error("Requested session not found", 404)

    if session_dir is None:
        return json_response({
            "session_dir": None,
            "session_name": None,
            "pending_action": None,
            "briefings": [],
            "summary": None,
        })

    try:
        summary = _load_json(session_dir / "session_summary.json")
        pending = _load_json(session_dir / "pending_action.json")
        briefings = _load_jsonl(session_dir / "agent_briefings.jsonl", limit)
    except RuntimeError as exc:
        return json_error(str(exc), 500)

    return json_response(
        {
            "session_dir": str(session_dir),
            "session_name": session_dir.name,
            "session_path": _relative_to_pipeline(session_dir),
            "pending_action": pending,
            "briefings": briefings,
            "summary": summary,
        }
    )


@csrf_exempt
async def post_selfplay_approval(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return json_error("Invalid JSON body", 400)
    try:
        payload = SelfplayApprovalRequest.model_validate(data)
    except ValidationError as exc:
        return json_response({"detail": exc.errors()}, status=422)

    try:
        session_dir = _resolve_session(payload.session)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except FileNotFoundError:
        return json_error("No self-play session found", 404)

    if session_dir is None:
        return json_error("No self-play session found", 404)

    pending_path = session_dir / "pending_action.json"
    try:
        data = _load_json(pending_path)
    except RuntimeError as exc:
        return json_error(str(exc), 500)
    if data is None:
        return json_error("pending_action.json not found for session", 404)

    status = payload.status.strip().lower()
    if not status:
        return json_error("Status must be non-empty", 400)

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
        return json_error(f"Failed to update pending action: {exc}", 500)

    return json_response({
        "session_dir": str(session_dir),
        "pending_action": data,
    })
