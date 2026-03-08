from __future__ import annotations

import asyncio
import io
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import queue as queue_mod
from queue import Queue
from typing import Any, Dict, Optional

from django.http import HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from pydantic import ValidationError

from ..job_controller import DashboardJobController, get_dashboard_jobs
from ..jobs import JobHandle
from ..pipeline_events import forward_pipeline_events
from ..pipeline_output import (
    line_to_pipeline_event as _line_to_event,
    resolve_run_dir_hint as _resolve_run_dir_hint,
)
from ..pipeline_runtime import (
    initialize_pipeline_job,
    launch_pipeline_job,
    PipelineLaunchRequest,
    multiprocessing_available as _runtime_multiprocessing_available,
    normalize_runner_mode as _runtime_normalize_runner_mode,
)
from ..subprocess_runtime import pump_text_subprocess_output
from ..helpers import (
    ROOT,
    build_pipeline_args,
    read_best_sharpe_from_run,
    resolve_config_path,
    resolve_dataset_preset,
    resolve_latest_run_dir,
)
from ..http import json_error, json_response
from ..models import PipelineRunRequest


JOB_STATE_RETENTION_SECONDS = 300.0


def _pipeline_worker(
    cli_args: list[str] | tuple[str, ...],
    root_dir: str | os.PathLike[str],
    queue_or_job_id: Any,
    *extra: Any,
) -> None:
    if not isinstance(cli_args, list):
        cli_args = list(cli_args)
    root_dir = os.fspath(root_dir)

    event_queue: Any | None = None
    aux_queues: list[Any] = []
    job_id: str | None = None
    extras: list[Any] = []

    def _is_queue(obj: Any) -> bool:
        put = getattr(obj, "put", None)
        return callable(put)

    for candidate in (queue_or_job_id, *extra):
        if _is_queue(candidate):
            if event_queue is None:
                event_queue = candidate
            else:
                aux_queues.append(candidate)
            continue
        if isinstance(candidate, (str, os.PathLike)) and job_id is None:
            job_id = os.fspath(candidate)
            continue
        if candidate is not None:
            extras.append(candidate)

    if event_queue is None:
        raise TypeError(
            "_pipeline_worker expected at least one queue argument for event dispatching"
        )
    if extras:
        logging.getLogger(__name__).warning(
            "Ignoring unexpected pipeline worker arguments: %r", extras
        )

    if job_id:
        os.environ.setdefault("PIPELINE_JOB_ID", job_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        os.chdir(root_dir)
    except Exception:
        pass

    def _push_event(line: str) -> None:
        try:
            event = _line_to_event(line)
            event_queue.put(event)
            for extra_q in aux_queues:
                try:
                    extra_q.put(event)
                except Exception:
                    pass
        except Exception:
            pass

    class _TeeStream(io.TextIOBase):
        def __init__(self, stream: io.TextIOBase | None) -> None:
            super().__init__()
            self._stream = stream
            self._buffer = ""
            self._encoding = getattr(stream, "encoding", "utf-8")
            self._errors = getattr(stream, "errors", "strict")

        def write(self, data: str) -> int:
            if not isinstance(data, str):
                data = str(data)
            if not data:
                return 0
            try:
                if self._stream is not None:
                    self._stream.write(data)
                    self._stream.flush()
            except Exception:
                pass
            normalized = data.replace("\r", "\n")
            self._buffer += normalized
            while True:
                idx = self._buffer.find("\n")
                if idx == -1:
                    break
                chunk = self._buffer[:idx]
                self._buffer = self._buffer[idx + 1 :]
                _push_event(chunk.rstrip("\r"))
            return len(data)

        def flush(self) -> None:
            try:
                if self._stream is not None:
                    self._stream.flush()
            except Exception:
                pass
            if self._buffer:
                _push_event(self._buffer.rstrip("\r"))
                self._buffer = ""

        def isatty(self) -> bool:
            if self._stream is None:
                return False
            try:
                return bool(self._stream.isatty())
            except Exception:
                return False

        @property
        def encoding(self) -> str:
            return self._encoding

        @property
        def errors(self) -> str:
            return self._errors

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = _TeeStream(original_stdout)
    tee_stderr = _TeeStream(original_stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    from alpha_evolve.cli.pipeline import parse_args, run_pipeline_programmatic

    try:
        from alpha_evolve.cli.pipeline import options_from_namespace  # type: ignore[attr-defined]
    except ImportError:
        options_from_namespace = None  # type: ignore
    from alpha_evolve.utils import logging as logging_setup

    original_setup = logging_setup.setup_logging

    try:
        evo_cfg, bt_cfg, ns = parse_args(cli_args)
        if options_from_namespace is None:
            from alpha_evolve.cli.pipeline import PipelineOptions

            options = PipelineOptions(
                debug_prints=getattr(ns, "debug_prints", False),
                run_baselines=getattr(ns, "run_baselines", False),
                retrain_baselines=getattr(ns, "retrain_baselines", False),
                log_level=getattr(ns, "log_level", "INFO"),
                log_file=getattr(ns, "log_file", None),
                dry_run=getattr(ns, "dry_run", False),
                output_dir=getattr(ns, "output_dir", None),
                persist_hof_per_gen=getattr(ns, "persist_hof_per_gen", True),
                disable_align_cache=getattr(ns, "disable_align_cache", False),
                align_cache_dir=getattr(ns, "align_cache_dir", None),
            )
        else:
            options = options_from_namespace(ns)

        run_dir = run_pipeline_programmatic(evo_cfg, bt_cfg, options)
        best = read_best_sharpe_from_run(run_dir) if run_dir.exists() else None
        event_queue.put(
            {
                "type": "final",
                "run_dir": str(run_dir.resolve()),
                "sharpe_best": None if best is None else float(best),
            }
        )
        event_queue.put({"type": "status", "msg": "exit", "code": 0})
    except Exception as exc:  # pragma: no cover - propagated back to UI
        event_queue.put({"type": "error", "code": 1, "detail": str(exc)})
        event_queue.put({"type": "status", "msg": "exit", "code": 1})
    finally:
        try:
            tee_stdout.flush()
            tee_stderr.flush()
        except Exception:
            pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logging_setup.setup_logging = original_setup
        event_queue.put({"type": "__complete__"})

def _build_subprocess_command(cli_args: list[str]) -> list[str]:
    """Return the command used for the sandbox-safe subprocess runner.

    Defined as a function so tests can monkeypatch it.
    """

    return [sys.executable, "-u", "-m", "alpha_evolve.cli.pipeline", *cli_args]


def _normalize_runner_mode(value: str | None) -> str:
    return _runtime_normalize_runner_mode(value)


def _multiprocessing_available() -> bool:
    return _runtime_multiprocessing_available(mp_context_getter=lambda: mp.get_context("spawn"))


def _pump_subprocess_output(
    *,
    job_id: str,
    proc: subprocess.Popen[str],
    event_queue: "queue_mod.Queue[dict[str, Any]]",
) -> None:
    run_dir_hint: Optional[str] = None

    def _handle_line(line: str) -> None:
        nonlocal run_dir_hint
        if run_dir_hint is None:
            run_dir_hint = _resolve_run_dir_hint(line)
        event_queue.put(_line_to_event(line))

    code = pump_text_subprocess_output(
        proc=proc,
        handle_line=_handle_line,
        handle_error=lambda exc: event_queue.put({"type": "error", "code": 1, "detail": str(exc)}),
    )

    run_dir: Optional[Path] = None
    if run_dir_hint:
        try:
            candidate = Path(run_dir_hint).expanduser()
            run_dir = (
                candidate.resolve()
                if candidate.is_absolute()
                else (ROOT / candidate).resolve()
            )
        except Exception:
            run_dir = None

    if code == 0 and run_dir is None:
        try:
            run_dir = resolve_latest_run_dir()
        except Exception:
            run_dir = None

    if code == 0 and run_dir is not None:
        best = read_best_sharpe_from_run(run_dir) if run_dir is not None else None
        event_queue.put(
            {
                "type": "final",
                "run_dir": str(run_dir.resolve()),
                "sharpe_best": None if best is None else float(best),
            }
        )

    event_queue.put({"type": "status", "msg": "exit", "code": int(code)})
    event_queue.put({"type": "__complete__"})


async def _forward_events(
    job_id: str,
    event_queue: Any,
    client_queue: Queue,
    controller: DashboardJobController | None = None,
) -> None:
    resolved_controller = controller or get_dashboard_jobs()
    await forward_pipeline_events(
        controller=resolved_controller,
        job_id=job_id,
        event_queue=event_queue,
        client_queue=client_queue,
        cleanup_delay_seconds=JOB_STATE_RETENTION_SECONDS,
    )


@csrf_exempt
async def start_pipeline_run(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    try:
        payload_data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return json_error("Invalid JSON body", 400)
    try:
        payload = PipelineRunRequest.model_validate(payload_data)
    except ValidationError as exc:
        return json_response({"detail": exc.errors()}, status=422)

    payload_dict = payload.model_dump()
    dataset = (payload_dict.get("dataset") or "").strip().lower()
    cfg_path = payload_dict.get("config")
    if cfg_path:
        resolved_cfg = resolve_config_path(str(cfg_path))
        if resolved_cfg is None:
            return json_error(f"Config not found: {cfg_path}", 404)
        payload_dict["config"] = str(resolved_cfg)
    elif dataset and not resolve_dataset_preset(dataset):
        return json_error(
            "Unknown dataset; use dataset=sp500, dataset=sp500_small, or provide a config path",
            400,
        )

    cli_args = build_pipeline_args(payload_dict, include_runner=False)
    full_args = build_pipeline_args(payload_dict, include_runner=True)
    controller = get_dashboard_jobs()
    job = initialize_pipeline_job(
        controller=controller,
        root_dir=ROOT,
        payload_dict=payload_dict,
        full_args=full_args,
    )

    try:
        launch = launch_pipeline_job(
            PipelineLaunchRequest(
                controller=controller,
                job=job,
                cli_args=cli_args,
                requested_mode_raw=payload_dict.get("runner_mode")
                or os.environ.get("AE_DASHBOARD_RUNNER_MODE"),
                root_dir=ROOT,
                pipeline_worker=_pipeline_worker,
                build_subprocess_command=_build_subprocess_command,
                pump_subprocess_output=_pump_subprocess_output,
                mp_context_getter=lambda: mp.get_context("spawn"),
            )
        )
    except ValueError as exc:
        controller.clear_job(job.job_id)
        return json_error(str(exc), 400)
    except Exception as exc:
        # Fail fast but keep a JSON error payload for the UI.
        controller.clear_job(job.job_id)
        return json_error(
            f"Failed to start pipeline runner: {exc}", 500
        )

    forward_task = asyncio.create_task(
        _forward_events(job.job_id, launch.event_queue, job.client_queue, controller=controller)
    )

    controller.set_handle(
        job.job_id,
        JobHandle(proc=launch.proc, task=forward_task, stop_cb=launch.stop_cb),
    )

    return json_response({"job_id": job.job_id})


def sse_events(request: HttpRequest, job_id: str):
    queue = get_dashboard_jobs().get_queue(job_id)
    if queue is None:
        return json_error("Unknown job id", 404)
    from ..helpers import make_sse_response

    return make_sse_response(queue)


@csrf_exempt
async def stop(request: HttpRequest, job_id: str):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    controller = get_dashboard_jobs()
    ok = controller.stop(job_id)
    if not ok:
        return json_error("Unknown job id or already stopped", 404)
    q = controller.get_queue(job_id)
    if q is not None:
        try:
            q.put_nowait(json.dumps({"type": "status", "msg": "stop_requested"}))
        except Exception:
            pass
    return json_response({"stopped": True})
