from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Dict

from django.http import HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from pydantic import ValidationError

from ..jobs import STATE, JobHandle
from ..helpers import (
    ROOT,
    build_pipeline_args,
    read_best_sharpe_from_run,
    resolve_dataset_preset,
    resolve_latest_run_dir,
    RE_SHARPE,
    RE_DIAG,
    RE_PROGRESS,
)
from ..http import json_error, json_response
from ..models import PipelineRunRequest


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _parse_constant(token: str) -> float:
    if token == "NaN":
        return float("nan")
    if token == "Infinity":
        return float("inf")
    if token == "-Infinity":
        return float("-inf")
    raise ValueError(f"Unexpected JSON constant: {token}")


def _sanitize_json_data(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_json_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_data(v) for v in value]
    return value


def _line_to_event(line: str) -> Dict[str, Any]:
    line = line.rstrip("\n")
    if line:
        line = ANSI_ESCAPE_RE.sub("", line)
    if not line:
        return {"type": "log", "raw": ""}
    if (m := RE_DIAG.search(line)) is not None:
        try:
            data = json.loads(m.group(1), parse_constant=_parse_constant)
            data = _sanitize_json_data(data)
            return {"type": "diag", "data": data, "raw": line}
        except Exception:
            return {"type": "log", "raw": line}
    if (m := RE_PROGRESS.search(line)) is not None:
        try:
            data = json.loads(m.group(1), parse_constant=_parse_constant)
            data = _sanitize_json_data(data)
            event = {"type": "progress", "data": data, "raw": line}
            if isinstance(data, dict) and isinstance(data.get("type"), str):
                event["subtype"] = data["type"]
            return event
        except Exception:
            return {"type": "log", "raw": line}
    if (m := RE_SHARPE.search(line)) is not None:
        try:
            return {"type": "score", "sharpe_best": float(m.group(1)), "raw": line}
        except Exception:
            return {"type": "log", "raw": line}
    return {"type": "log", "raw": line}


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
        raise TypeError("_pipeline_worker expected at least one queue argument for event dispatching")
    if extras:
        logging.getLogger(__name__).warning("Ignoring unexpected pipeline worker arguments: %r", extras)

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
        latest = resolve_latest_run_dir()
        best = read_best_sharpe_from_run(latest) if latest is not None else None
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


_PIPELINE_WORKER_DEFAULT = _pipeline_worker


async def _forward_events(
    job_id: str,
    event_queue: mp.Queue,
    client_queue: Queue,
) -> None:
    loop = asyncio.get_running_loop()
    log_handle = None

    def _log_line(line: str) -> None:
        nonlocal log_handle
        if not isinstance(line, str):
            return
        activity = STATE.get_activity(job_id) or {}
        log_path = activity.get("log_path")
        if not isinstance(log_path, str):
            return
        try:
            if log_handle is None:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                log_handle = open(log_path, "a", encoding="utf-8")
            if line.endswith("\n"):
                log_handle.write(line)
            else:
                log_handle.write(line + "\n")
            log_handle.flush()
        except Exception:
            pass

    def _touch_activity(**updates: Any) -> None:
        STATE.update_activity(job_id, updated_at=time.time(), **updates)

    try:
        while True:
            item = await loop.run_in_executor(None, event_queue.get)
            if not isinstance(item, dict):
                break
            event_type = item.get("type")
            raw_line = item.get("raw") if isinstance(item, dict) else None
            if event_type == "__complete__":
                break
            _touch_activity()
            if event_type == "log":
                if isinstance(raw_line, str):
                    text = raw_line.strip()
                    if text:
                        _touch_activity(last_message=text)
            elif event_type == "progress":
                data = item.get("data")
                subtype = item.get("subtype") or (data.get("type") if isinstance(data, dict) else None)
                if subtype == "gen_progress" and isinstance(data, dict):
                    _touch_activity(progress=data)
                elif subtype == "gen_summary" and isinstance(data, dict):
                    meta = STATE.meta.get(job_id)
                    if isinstance(meta, dict):
                        history = meta.setdefault("gen_history", [])
                        history.append(data)
                        # keep history bounded to avoid runaway memory use
                        if len(history) > 2000:
                            del history[0 : len(history) - 2000]
                    STATE.append_activity_summary(job_id, data)
                    _touch_activity(progress=data)
            elif event_type == "score":
                sharpe = item.get("sharpe_best")
                try:
                    value = float(sharpe)
                except (TypeError, ValueError):
                    value = None
                if value is not None:
                    _touch_activity(sharpe_best=value)
            elif event_type == "status":
                msg = item.get("msg")
                if msg == "exit":
                    try:
                        code = int(item.get("code", 1))
                    except Exception:
                        code = 1
                    success = code == 0
                    _touch_activity(
                        status="complete" if success else "error",
                        last_message="Pipeline finished." if success else "Pipeline stopped.",
                    )
                    if not success:
                        STATE.pop_meta(job_id)
                elif isinstance(msg, str):
                    mapped = "Pipeline started." if msg == "started" else msg
                    _touch_activity(status="running", last_message=mapped)
            elif event_type == "error":
                detail = item.get("detail")
                if isinstance(detail, str) and detail.strip():
                    message = detail
                    _log_line(detail)
                else:
                    message = "Pipeline error."
                _touch_activity(status="error", last_message=message)
            elif event_type == "final":
                context = STATE.pop_meta(job_id)
                history = None
                if isinstance(context, dict):
                    history = context.pop("gen_history", None)
                if context:
                    try:
                        run_path = Path(item["run_dir"]).resolve()
                        meta_dir = run_path / "meta"
                        meta_dir.mkdir(exist_ok=True)
                        context_out = dict(context)
                        context_out["run_dir"] = str(run_path)
                        with open(meta_dir / "ui_context.json", "w", encoding="utf-8") as fh:
                            json.dump(context_out, fh, indent=2)
                        if history:
                            with open(meta_dir / "gen_summary.jsonl", "w", encoding="utf-8") as fh_hist:
                                for entry in history:
                                    fh_hist.write(json.dumps(entry))
                                    fh_hist.write("\n")
                    except Exception:
                        pass
                sharpe = item.get("sharpe_best")
                try:
                    value = float(sharpe)
                except (TypeError, ValueError):
                    value = None
                if value is not None:
                    _touch_activity(sharpe_best=value)
                _touch_activity(status="complete")
            if isinstance(raw_line, str):
                STATE.add_log(job_id, raw_line)
                if raw_line:
                    _log_line(raw_line)
            try:
                client_queue.put_nowait(json.dumps(item))
            except Exception:
                pass
    finally:
        STATE.clear_handle(job_id)
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass


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
        path_obj = Path(str(cfg_path))
        if not path_obj.exists():
            return json_error(f"Config not found: {path_obj}", 404)
    elif dataset and not resolve_dataset_preset(dataset):
        return json_error("Unknown dataset; use dataset=sp500, dataset=sp500_small, or provide a config path", 400)

    cli_args = build_pipeline_args(payload_dict, include_runner=False)
    full_args = build_pipeline_args(payload_dict, include_runner=True)

    import uuid as _uuid

    job_id = str(_uuid.uuid4())
    client_queue = STATE.new_queue(job_id)

    submitted_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    ui_context = {
        "job_id": job_id,
        "submitted_at": submitted_at,
        "payload": payload_dict,
        "pipeline_args": full_args,
    }
    STATE.set_meta(job_id, ui_context)
    STATE.init_activity(
        job_id,
        {
            "status": "running",
            "last_message": "Pipeline started.",
            "sharpe_best": None,
            "progress": None,
            "summaries": [],
            "updated_at": time.time(),
        },
    )

    client_queue.put_nowait(json.dumps({"type": "status", "msg": "started", "args": full_args}))

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"pipeline_{job_id}.log"
    STATE.update_activity(job_id, log_path=str(log_file_path))

    ctx = mp.get_context("spawn")
    event_queue: mp.Queue = ctx.Queue()
    worker_fn = _pipeline_worker
    worker_args: tuple[Any, ...] = (cli_args, str(ROOT), event_queue)
    if worker_fn is _PIPELINE_WORKER_DEFAULT:
        worker_args += (job_id,)
    proc = ctx.Process(target=worker_fn, args=worker_args)
    proc.start()

    forward_task = asyncio.create_task(_forward_events(job_id, event_queue, client_queue))

    def _stop() -> None:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
        try:
            event_queue.put_nowait({"type": "__complete__"})
        except Exception:
            pass
        forward_task.cancel()

    STATE.set_handle(job_id, JobHandle(proc=proc, task=forward_task, stop_cb=_stop))

    return json_response({"job_id": job_id})

def sse_events(request: HttpRequest, job_id: str):
    queue = STATE.get_queue(job_id)
    if queue is None:
        return json_error("Unknown job id", 404)
    from ..helpers import make_sse_response

    return make_sse_response(queue)


@csrf_exempt
async def stop(request: HttpRequest, job_id: str):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    ok = STATE.stop(job_id)
    if not ok:
        return json_error("Unknown job id or already stopped", 404)
    return json_response({"stopped": True})
