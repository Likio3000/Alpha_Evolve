from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .job_controller import DashboardJobController
from .subprocess_runtime import start_text_subprocess
from .subprocess_runtime import pump_text_subprocess_output


@dataclass(frozen=True)
class MLLabLaunchRequest:
    controller: DashboardJobController
    root_dir: Path
    run_dir: Path
    run_dir_label: str
    spec_payload: dict[str, Any]
    spec_path: Path
    progress_re: Any
    cleanup_delay_seconds: float


@dataclass(frozen=True)
class MLLabLaunch:
    job_id: str
    run_dir: Path
    run_dir_label: str
    spec_path: Path
    log_path: Path
    proc: subprocess.Popen[str]


def parse_progress_line(line: str, progress_re) -> dict[str, Any] | None:
    match = progress_re.search(line)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def update_activity_from_progress(
    *,
    controller: DashboardJobController,
    job_id: str,
    payload: dict[str, Any],
) -> None:
    now = time.time()
    activity = controller.get_activity(job_id) or {}
    progress = dict(activity.get("progress") or {})
    progress.update(payload)
    updates: dict[str, Any] = {"progress": progress, "updated_at": now}
    message = None
    if payload.get("type") == "ml_model_start":
        label = payload.get("model_label") or payload.get("model_id")
        variant = payload.get("variant")
        message = f"Training {label} ({variant})" if variant else f"Training {label}"
    elif payload.get("type") == "ml_model_end":
        label = payload.get("model_label") or payload.get("model_id")
        variant = payload.get("variant")
        message = f"Finished {label} ({variant})" if variant else f"Finished {label}"
        sharpe = payload.get("sharpe")
        try:
            sharpe_value = float(sharpe)
        except Exception:
            sharpe_value = None
        if sharpe_value is not None:
            current = activity.get("sharpe_best")
            if current is None or sharpe_value > float(current):
                updates["sharpe_best"] = sharpe_value
    elif payload.get("type") == "ml_complete":
        message = "ML run complete."
    elif payload.get("type") == "ml_error":
        message = payload.get("message") or "ML run error."
        updates["status"] = "error"
    if message:
        updates["last_message"] = message
    controller.touch_activity(job_id, **updates)


def pump_ml_lab_subprocess_output(
    *,
    controller: DashboardJobController,
    job_id: str,
    proc: subprocess.Popen[str],
    log_path: Path,
    cleanup_delay_seconds: float,
    progress_re,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exit_code = 1
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            def _handle_line(raw_line: str) -> None:
                line = raw_line.rstrip("\n")
                log_file.write(line + "\n")
                log_file.flush()
                controller.add_log(job_id, line)
                payload = parse_progress_line(line, progress_re)
                if payload:
                    update_activity_from_progress(
                        controller=controller,
                        job_id=job_id,
                        payload=payload,
                    )

            exit_code = pump_text_subprocess_output(
                proc=proc,
                handle_line=_handle_line,
                wait_timeout=1.0,
            )
    finally:
        status = "complete" if exit_code == 0 else "error"
        controller.touch_activity(
            job_id,
            status=status,
            last_message="ML run finished." if exit_code == 0 else "ML run failed.",
            updated_at=time.time(),
        )
        controller.clear_handle(job_id)
        controller.schedule_cleanup(job_id, delay_seconds=cleanup_delay_seconds)


def launch_ml_lab_job(
    request: MLLabLaunchRequest,
    *,
    now_time: Callable[[], float] = time.time,
) -> MLLabLaunch:
    request.run_dir.mkdir(parents=True, exist_ok=True)
    request.spec_path.write_text(
        json.dumps(request.spec_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    job_id, _client_queue = request.controller.initialize_job(
        job_id=str(uuid.uuid4()),
        activity={
            "status": "running",
            "last_message": "ML run started.",
            "progress": None,
            "updated_at": now_time(),
            "run_dir": request.run_dir_label,
        },
    )

    log_dir = request.root_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ml_lab_{job_id}.log"
    request.controller.set_log_path(job_id, str(log_path))

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "alpha_evolve.cli.ml_lab",
        "--spec",
        str(request.spec_path),
        "--out_dir",
        str(request.run_dir),
    ]

    env = os.environ.copy()
    python_paths = [str(request.root_dir / "src"), str(request.root_dir)]
    existing = env.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    proc = start_text_subprocess(cmd=cmd, cwd=request.root_dir, env=env)
    request.controller.set_proc(job_id, proc)
    thread = threading.Thread(
        target=pump_ml_lab_subprocess_output,
        kwargs={
            "controller": request.controller,
            "job_id": job_id,
            "proc": proc,
            "log_path": log_path,
            "cleanup_delay_seconds": request.cleanup_delay_seconds,
            "progress_re": request.progress_re,
        },
        daemon=True,
    )
    thread.start()
    return MLLabLaunch(
        job_id=job_id,
        run_dir=request.run_dir,
        run_dir_label=request.run_dir_label,
        spec_path=request.spec_path,
        log_path=log_path,
        proc=proc,
    )
