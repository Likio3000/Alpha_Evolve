from __future__ import annotations

import json
import os
import queue as queue_mod
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Callable

from .job_controller import DashboardJobController
from .subprocess_runtime import start_text_subprocess


@dataclass(frozen=True)
class PipelineLaunch:
    job_id: str
    client_queue: Queue
    event_queue: Any
    proc: Any
    requested_mode: str
    resolved_mode: str
    stop_cb: Callable[[], None]


@dataclass(frozen=True)
class PipelineJobContext:
    job_id: str
    client_queue: Queue
    log_path: Path


@dataclass(frozen=True)
class PipelineLaunchRequest:
    controller: DashboardJobController
    job: PipelineJobContext
    cli_args: list[str]
    requested_mode_raw: str | None
    root_dir: Path
    pipeline_worker: Callable[..., None]
    build_subprocess_command: Callable[[list[str]], list[str]]
    pump_subprocess_output: Callable[..., None]
    mp_context_getter: Callable[[], Any]


def normalize_runner_mode(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return "auto"
    if raw in {"auto", "default"}:
        return "auto"
    if raw in {"mp", "multiprocessing", "process", "proc"}:
        return "multiprocessing"
    if raw in {"subprocess", "subproc", "spawn"}:
        return "subprocess"
    return raw


def multiprocessing_available(*, mp_context_getter: Callable[[], Any]) -> bool:
    try:
        ctx = mp_context_getter()
        q = ctx.Queue()
        try:
            q.put_nowait({"type": "__probe__"})
        except Exception:
            q.put({"type": "__probe__"})
        close = getattr(q, "close", None)
        if callable(close):
            close()
        join_thread = getattr(q, "join_thread", None)
        if callable(join_thread):
            try:
                join_thread()
            except Exception:
                pass
        return True
    except Exception:
        return False


def initialize_pipeline_job(
    *,
    controller: DashboardJobController,
    root_dir: Path,
    payload_dict: dict[str, Any],
    full_args: list[str],
    now_time: Callable[[], float] = time.time,
    now_datetime: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> PipelineJobContext:
    job_id = str(uuid.uuid4())

    submitted_at = now_datetime().isoformat().replace("+00:00", "Z")
    ui_context = {
        "job_id": job_id,
        "submitted_at": submitted_at,
        "payload": payload_dict,
        "pipeline_args": full_args,
    }
    job_id, client_queue = controller.initialize_job(
        job_id=job_id,
        meta=ui_context,
        activity={
            "status": "running",
            "last_message": "Pipeline started.",
            "sharpe_best": None,
            "progress": None,
            "summaries": [],
            "updated_at": now_time(),
        },
    )

    client_queue.put_nowait(json.dumps({"type": "status", "msg": "started", "args": full_args}))

    log_dir = root_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"pipeline_{job_id}.log"
    controller.set_log_path(job_id, str(log_file_path))
    return PipelineJobContext(
        job_id=job_id,
        client_queue=client_queue,
        log_path=log_file_path,
    )


def _start_subprocess_runner(
    *,
    job_id: str,
    cli_args: list[str],
    root_dir: Path,
    build_subprocess_command: Callable[[list[str]], list[str]],
    pump_subprocess_output: Callable[..., None],
) -> tuple[Any, queue_mod.Queue[dict[str, Any]]]:
    q: queue_mod.Queue[dict[str, Any]] = queue_mod.Queue()
    cmd = build_subprocess_command(cli_args)
    env = os.environ.copy()
    env.setdefault("PIPELINE_JOB_ID", job_id)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = start_text_subprocess(cmd=cmd, cwd=root_dir, env=env)
    thread = threading.Thread(
        target=pump_subprocess_output,
        kwargs={"job_id": job_id, "proc": proc, "event_queue": q},
        daemon=True,
    )
    thread.start()
    return proc, q


def launch_pipeline_job(
    request: PipelineLaunchRequest,
    now_time: Callable[[], float] = time.time,
) -> PipelineLaunch:
    controller = request.controller
    job_id = request.job.job_id
    requested_mode = normalize_runner_mode(request.requested_mode_raw)
    resolved_mode = requested_mode
    if requested_mode == "auto":
        resolved_mode = (
            "multiprocessing"
            if multiprocessing_available(mp_context_getter=request.mp_context_getter)
            else "subprocess"
        )
    if resolved_mode not in {"multiprocessing", "subprocess"}:
        raise ValueError("runner_mode must be one of: auto, multiprocessing, subprocess")

    controller.touch_activity(job_id, runner_mode=resolved_mode)

    if resolved_mode == "multiprocessing":
        try:
            ctx = request.mp_context_getter()
            event_queue = ctx.Queue()
            worker_args: tuple[Any, ...] = (request.cli_args, str(request.root_dir), event_queue)
            if getattr(request.pipeline_worker, "__name__", "") == "_pipeline_worker":
                worker_args += (job_id,)
            proc = ctx.Process(target=request.pipeline_worker, args=worker_args)
            proc.start()
        except Exception as exc:
            if requested_mode != "auto":
                raise
            controller.touch_activity(
                job_id,
                runner_mode="subprocess",
                last_message=f"Falling back to subprocess runner ({exc}).",
            )
            resolved_mode = "subprocess"
            proc, event_queue = _start_subprocess_runner(
                job_id=job_id,
                cli_args=request.cli_args,
                root_dir=request.root_dir,
                build_subprocess_command=request.build_subprocess_command,
                pump_subprocess_output=request.pump_subprocess_output,
            )
    else:
        proc, event_queue = _start_subprocess_runner(
            job_id=job_id,
            cli_args=request.cli_args,
            root_dir=request.root_dir,
            build_subprocess_command=request.build_subprocess_command,
            pump_subprocess_output=request.pump_subprocess_output,
        )

    def _mark_stop_requested() -> None:
        controller.touch_activity(job_id, updated_at=now_time(), last_message="Stop requested…")

    def _stop() -> None:
        _mark_stop_requested()
        try:
            if resolved_mode == "multiprocessing":
                if getattr(proc, "is_alive", lambda: False)():
                    proc.terminate()
                try:
                    event_queue.put_nowait({"type": "status", "msg": "exit", "code": 1})
                    event_queue.put_nowait({"type": "__complete__"})
                except Exception:
                    pass
                return
            if hasattr(proc, "poll") and proc.poll() is None:
                proc.terminate()

                def _kill_after_timeout(p: Any) -> None:
                    time.sleep(5)
                    try:
                        if p.poll() is None:
                            p.kill()
                    except Exception:
                        pass

                threading.Thread(target=_kill_after_timeout, args=(proc,), daemon=True).start()
        except Exception:
            pass

    return PipelineLaunch(
        job_id=job_id,
        client_queue=request.job.client_queue,
        event_queue=event_queue,
        proc=proc,
        requested_mode=requested_mode,
        resolved_mode=resolved_mode,
        stop_cb=_stop,
    )
