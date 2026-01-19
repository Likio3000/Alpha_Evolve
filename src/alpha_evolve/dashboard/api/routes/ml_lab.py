from __future__ import annotations

import json
import random
import re
import subprocess
import threading
import time
import uuid
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.http import HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from alpha_evolve.ml_lab.models import available_models_payload

from ..helpers import PIPELINE_DIR, ROOT, RE_PROGRESS, resolve_dataset_preset
from ..http import json_error, json_response
from ..jobs import STATE


ML_RUNS_DIR = PIPELINE_DIR / "ml_runs"

_CODENAME_RNG = random.SystemRandom()
_CODENAME_FIRST_NAMES = [
    # Spanish
    "Lucia",
    "Mateo",
    "Sofia",
    "Diego",
    "Ines",
    "Carlos",
    # English
    "Harper",
    "Ethan",
    "Ava",
    "Noah",
    "Amelia",
    "Oliver",
    # Polish
    "Zofia",
    "Jakub",
    "Maja",
    "Leon",
    "Ania",
    "Oskar",
    # French
    "Hugo",
    "Manon",
    "Leo",
    "Camille",
    "Lucas",
    "Chloe",
]
_CODENAME_LAST_NAMES = [
    "Garcia",
    "Lopez",
    "Martinez",
    "Santos",
    "Hernandez",
    "Ramirez",
    "Smith",
    "Johnson",
    "Bennett",
    "Clark",
    "Baker",
    "Turner",
    "Kowalski",
    "Nowak",
    "Wisniewski",
    "Lewandowski",
    "Mazur",
    "Kaminski",
    "Dubois",
    "Lefevre",
    "Moreau",
    "Rousseau",
    "Leroux",
    "Boulanger",
]
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _generate_run_codename() -> str:
    first = _CODENAME_RNG.choice(_CODENAME_FIRST_NAMES)
    last = _CODENAME_RNG.choice(_CODENAME_LAST_NAMES)
    return f"{first}-{last}"


def _slugify_label(value: str) -> str:
    cleaned = _SLUG_RE.sub("_", value.strip()).strip("_")
    return cleaned or "custom"


def _format_seed(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 42


def _ensure_runs_dir() -> Path:
    ML_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return ML_RUNS_DIR


def _safe_read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _format_path_for_ui(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        try:
            return str(p.relative_to(PIPELINE_DIR))
        except ValueError:
            return str(p)


def _find_runs() -> List[Path]:
    runs = [p for p in _ensure_runs_dir().glob("*") if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _resolve_run_dir(run_dir: str) -> Path:
    candidate = Path(run_dir)
    base = _ensure_runs_dir().resolve()
    root = ROOT.resolve()

    def _is_under(path: Path) -> bool:
        try:
            path.relative_to(base)
        except ValueError:
            return False
        return True

    resolved: Path | None = None
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved_root = (root / candidate).resolve()
        if _is_under(resolved_root):
            resolved = resolved_root
        else:
            resolved_base = (base / candidate).resolve()
            if _is_under(resolved_base):
                resolved = resolved_base

    if resolved is None or not _is_under(resolved):
        raise ValueError("run_dir must resolve under pipeline_runs_cs/ml_runs")
    if not resolved.exists():
        raise ValueError("run_dir not found")
    return resolved


def list_models(_request: HttpRequest):
    return json_response({"models": available_models_payload()})


def list_runs(request: HttpRequest):
    limit_param = request.GET.get("limit", "50")
    try:
        limit = max(1, min(1000, int(limit_param)))
    except Exception:
        return json_error("limit must be an integer", 400)
    items: List[Dict[str, Any]] = []
    for p in _find_runs()[:limit]:
        summary = _safe_read_json(p / "ml_summary.json") or {}
        meta = _safe_read_json(p / "meta" / "run_metadata.json") or {}
        items.append(
            {
                "path": _format_path_for_ui(p),
                "name": p.name,
                "status": meta.get("status"),
                "best_sharpe": summary.get("best_sharpe"),
                "completed": summary.get("completed"),
                "total": summary.get("total"),
                "started_at": meta.get("started_at"),
            }
        )
    return json_response(items)


def run_details(request: HttpRequest):
    run_dir = request.GET.get("run_dir")
    if not run_dir:
        return json_error("run_dir is required", 400)
    try:
        resolved = _resolve_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)
    summary = _safe_read_json(resolved / "ml_summary.json")
    results = _safe_read_json(resolved / "ml_results.json")
    spec = _safe_read_json(resolved / "ml_spec.json")
    meta = _safe_read_json(resolved / "meta" / "run_metadata.json")
    return json_response(
        {
            "path": _format_path_for_ui(resolved),
            "name": resolved.name,
            "summary": summary,
            "results": results,
            "spec": spec,
            "meta": meta,
        }
    )


def _parse_progress_line(line: str) -> Optional[Dict[str, Any]]:
    match = RE_PROGRESS.search(line)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def _update_activity_from_progress(job_id: str, payload: Dict[str, Any]) -> None:
    now = time.time()
    activity = STATE.get_activity(job_id) or {}
    progress = dict(activity.get("progress") or {})
    progress.update(payload)
    updates: Dict[str, Any] = {"progress": progress, "updated_at": now}
    msg = None
    if payload.get("type") == "ml_model_start":
        label = payload.get("model_label") or payload.get("model_id")
        variant = payload.get("variant")
        msg = f"Training {label} ({variant})" if variant else f"Training {label}"
    elif payload.get("type") == "ml_model_end":
        label = payload.get("model_label") or payload.get("model_id")
        variant = payload.get("variant")
        msg = f"Finished {label} ({variant})" if variant else f"Finished {label}"
        sharpe = payload.get("sharpe")
        try:
            sharpe_val = float(sharpe)
        except Exception:
            sharpe_val = None
        if sharpe_val is not None:
            current = activity.get("sharpe_best")
            if current is None or sharpe_val > float(current):
                updates["sharpe_best"] = sharpe_val
    elif payload.get("type") == "ml_complete":
        msg = "ML run complete."
    elif payload.get("type") == "ml_error":
        msg = payload.get("message") or "ML run error."
        updates["status"] = "error"
    if msg:
        updates["last_message"] = msg
    STATE.update_activity(job_id, **updates)


def _pump_subprocess_output(job_id: str, proc: subprocess.Popen, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                log_file.write(line + "\n")
                log_file.flush()
                STATE.add_log(job_id, line)
                payload = _parse_progress_line(line)
                if payload:
                    _update_activity_from_progress(job_id, payload)
    finally:
        exit_code = None
        try:
            exit_code = proc.wait(timeout=1)
        except Exception:
            pass
        status = "complete" if exit_code == 0 else "error"
        STATE.update_activity(
            job_id,
            status=status,
            last_message="ML run finished." if exit_code == 0 else "ML run failed.",
            updated_at=time.time(),
        )
        STATE.clear_handle(job_id)


@csrf_exempt
def start_run(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return json_error("Invalid JSON body", 400)
    if not isinstance(payload, dict):
        return json_error("Payload must be a JSON object", 400)

    dataset = str(payload.get("dataset") or "").strip().lower()
    cfg_path = payload.get("config")
    data_dir = payload.get("data_dir")

    if cfg_path:
        cfg_path = str(cfg_path)
        if not Path(cfg_path).exists():
            return json_error(f"Config not found: {cfg_path}", 404)
        payload["config"] = cfg_path
    elif dataset:
        preset = resolve_dataset_preset(dataset)
        if preset is None:
            return json_error(
                "Unknown dataset; use dataset=sp500, dataset=sp500_small, or provide a config path",
                400,
            )
        payload["config"] = str(preset)
    elif not data_dir:
        return json_error("config, dataset, or data_dir is required", 400)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    codename = _generate_run_codename()
    seed_label = _format_seed(payload.get("seed"))
    dataset_label = dataset or ""
    if not dataset_label and cfg_path:
        dataset_label = Path(cfg_path).stem
    dataset_label = _slugify_label(dataset_label)
    run_dir = (
        _ensure_runs_dir()
        / f"run_ml_{codename}_seed{seed_label}_{dataset_label}_{run_stamp}"
    )
    attempts = 0
    while run_dir.exists():
        attempts += 1
        codename = f"{_generate_run_codename()}-{attempts}"
        run_dir = (
            _ensure_runs_dir()
            / f"run_ml_{codename}_seed{seed_label}_{dataset_label}_{run_stamp}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    spec_path = run_dir / "ml_spec.json"
    spec_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "alpha_evolve.cli.ml_lab",
        "--spec",
        str(spec_path),
        "--out_dir",
        str(run_dir),
    ]

    job_id = str(uuid.uuid4())
    STATE.new_queue(job_id)
    STATE.init_activity(
        job_id,
        {
            "status": "running",
            "last_message": "ML run started.",
            "progress": None,
            "updated_at": time.time(),
            "run_dir": _format_path_for_ui(run_dir),
        },
    )

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ml_lab_{job_id}.log"
    STATE.update_activity(job_id, log_path=str(log_path))

    env = os.environ.copy()
    python_paths = [str(ROOT / "src"), str(ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    STATE.set_proc(job_id, proc)
    thread = threading.Thread(
        target=_pump_subprocess_output,
        kwargs={"job_id": job_id, "proc": proc, "log_path": log_path},
        daemon=True,
    )
    thread.start()

    return json_response({"job_id": job_id, "run_dir": _format_path_for_ui(run_dir)})


@csrf_exempt
def stop_run(request: HttpRequest, job_id: str):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    stopped = STATE.stop(job_id)
    if stopped:
        STATE.update_activity(
            job_id,
            last_message="Stop requested.",
            updated_at=time.time(),
        )
    return json_response({"stopped": bool(stopped)})
