from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
from datetime import datetime
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
    resolve_latest_run_dir,
    RE_SHARPE,
    RE_DIAG,
    RE_PROGRESS,
)
from ..http import json_error, json_response
from ..models import PipelineRunRequest


def _line_to_event(line: str) -> Dict[str, Any]:
    line = line.rstrip("\n")
    if not line:
        return {"type": "log", "raw": ""}
    if (m := RE_DIAG.search(line)) is not None:
        try:
            return {"type": "diag", "data": json.loads(m.group(1)), "raw": line}
        except Exception:
            return {"type": "log", "raw": line}
    if (m := RE_PROGRESS.search(line)) is not None:
        try:
            return {"type": "progress", "data": json.loads(m.group(1)), "raw": line}
        except Exception:
            return {"type": "log", "raw": line}
    if (m := RE_SHARPE.search(line)) is not None:
        try:
            return {"type": "score", "sharpe_best": float(m.group(1)), "raw": line}
        except Exception:
            return {"type": "log", "raw": line}
    return {"type": "log", "raw": line}


def _pipeline_worker(cli_args: list[str], root_dir: str, event_queue: mp.Queue) -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        os.chdir(root_dir)
    except Exception:
        pass

    from run_pipeline import parse_args, run_pipeline_programmatic
    try:
        from run_pipeline import options_from_namespace  # type: ignore[attr-defined]
    except ImportError:
        options_from_namespace = None  # type: ignore
    from utils import logging_setup

    class _QueueHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                event_queue.put(_line_to_event(record.getMessage()))
            except Exception:
                pass

    queue_handler = _QueueHandler(level=logging.INFO)

    original_setup = logging_setup.setup_logging

    def setup_logging_wrapper(level: int = logging.INFO, log_file: str | None = None) -> None:
        original_setup(level=level, log_file=log_file)
        logging.getLogger().addHandler(queue_handler)

    logging_setup.setup_logging = setup_logging_wrapper

    try:
        evo_cfg, bt_cfg, ns = parse_args(cli_args)
        if options_from_namespace is None:
            from run_pipeline import PipelineOptions

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
        logging_setup.setup_logging = original_setup
        event_queue.put({"type": "__complete__"})


async def _forward_events(
    job_id: str,
    event_queue: mp.Queue,
    client_queue: Queue,
) -> None:
    loop = asyncio.get_running_loop()
    try:
        while True:
            item = await loop.run_in_executor(None, event_queue.get)
            if not isinstance(item, dict):
                break
            event_type = item.get("type")
            if event_type == "__complete__":
                break
            if event_type == "log":
                STATE.add_log(job_id, item.get("raw", ""))
            elif event_type == "final":
                context = STATE.pop_meta(job_id)
                if context:
                    try:
                        run_path = Path(item["run_dir"]).resolve()
                        meta_dir = run_path / "meta"
                        meta_dir.mkdir(exist_ok=True)
                        context_out = dict(context)
                        context_out["run_dir"] = str(run_path)
                        with open(meta_dir / "ui_context.json", "w", encoding="utf-8") as fh:
                            json.dump(context_out, fh, indent=2)
                    except Exception:
                        pass
            elif event_type == "status" and item.get("msg") == "exit" and item.get("code") != 0:
                STATE.pop_meta(job_id)
            try:
                client_queue.put_nowait(json.dumps(item))
            except Exception:
                pass
    finally:
        STATE.clear_handle(job_id)


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
    elif dataset and dataset not in {"sp500", "s&p500", "snp500"}:
        return json_error("Unknown dataset; provide dataset=sp500 or a config path", 400)

    cli_args = build_pipeline_args(payload_dict, include_runner=False)
    full_args = build_pipeline_args(payload_dict, include_runner=True)

    import uuid as _uuid

    job_id = str(_uuid.uuid4())
    client_queue = STATE.new_queue(job_id)

    ui_context = {
        "job_id": job_id,
        "submitted_at": datetime.utcnow().isoformat() + "Z",
        "payload": payload_dict,
        "pipeline_args": full_args,
    }
    STATE.set_meta(job_id, ui_context)

    client_queue.put_nowait(json.dumps({"type": "status", "msg": "started", "args": full_args}))

    ctx = mp.get_context("spawn")
    event_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_pipeline_worker, args=(cli_args, str(ROOT), event_queue), daemon=True)
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
