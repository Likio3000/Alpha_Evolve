from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from django.http import HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from pydantic import ValidationError

from ..jobs import STATE
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

    pd = payload.model_dump()
    ds = (pd.get("dataset") or "").strip().lower()
    cfg_path = pd.get("config")
    if cfg_path:
        p = Path(str(cfg_path))
        if not p.exists():
            return json_error(f"Config not found: {p}", 404)
    elif ds and ds not in ("crypto", "crypto_4h", "crypto4h", "sp500", "s&p500", "snp500"):
        return json_error("Unknown dataset; provide dataset=crypto|sp500 or a config path", 400)

    import uuid as _uuid

    job_id = str(_uuid.uuid4())
    q = STATE.new_queue(job_id)

    args = build_pipeline_args(payload.model_dump())
    context_payload = payload.model_dump()
    ui_context = {
        "job_id": job_id,
        "submitted_at": datetime.utcnow().isoformat() + "Z",
        "payload": context_payload,
        "pipeline_args": args,
    }
    STATE.set_meta(job_id, ui_context)

    async def _pump():
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            total_cpus = os.cpu_count() or 1
            if total_cpus > 1:
                worker_cpus = max(1, total_cpus - 1)
                for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env.setdefault(var, str(worker_cpus))
        except Exception:
            pass

        def _prep_child() -> None:
            try:
                if hasattr(os, "nice"):
                    os.nice(5)
            except Exception:
                pass
            try:
                if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
                    current = os.sched_getaffinity(0)
                    if len(current) > 1:
                        keep = sorted(current)[1:]
                        if keep:
                            os.sched_setaffinity(0, set(keep))
            except Exception:
                pass

        proc = subprocess.Popen(
            args,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            preexec_fn=_prep_child,
        )

        STATE.set_proc(job_id, proc)
        q.put_nowait(json.dumps({"type": "status", "msg": "started", "args": args}))

        loop = asyncio.get_running_loop()
        re_sharpe = RE_SHARPE
        re_diag = RE_DIAG
        re_progress = RE_PROGRESS

        def _stream_output() -> None:
            def _put(item: dict[str, Any]) -> None:
                loop.call_soon_threadsafe(q.put_nowait, json.dumps(item))

            def _log(line: str) -> None:
                loop.call_soon_threadsafe(STATE.add_log, job_id, line)

            last_line: str | None = None

            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    if line:
                        last_line = line
                    try:
                        print(line, flush=True)
                    except Exception:
                        pass
                    _log(line)
                    md = re_diag.search(line)
                    if md:
                        try:
                            diag_obj = json.loads(md.group(1))
                            _put({"type": "diag", "data": diag_obj})
                        except Exception:
                            _put({"type": "log", "raw": line})
                        continue
                    mp = re_progress.search(line)
                    if mp:
                        try:
                            prog_obj = json.loads(mp.group(1))
                            _put({"type": "progress", "data": prog_obj})
                        except Exception:
                            _put({"type": "log", "raw": line})
                        continue
                    ms = re_sharpe.search(line)
                    if ms:
                        _put({"type": "score", "sharpe_best": float(ms.group(1)), "raw": line})
                    else:
                        _put({"type": "log", "raw": line})
            finally:
                code = proc.wait()
                latest = resolve_latest_run_dir()
                context = STATE.pop_meta(job_id)
                best = None
                if latest is not None:
                    best = read_best_sharpe_from_run(latest)
                    if context:
                        try:
                            run_path = latest.resolve()
                            meta_dir = run_path / "meta"
                            meta_dir.mkdir(exist_ok=True)
                            context_out = dict(context)
                            context_out["run_dir"] = str(run_path)
                            with open(meta_dir / "ui_context.json", "w", encoding="utf-8") as fh:
                                json.dump(context_out, fh, indent=2)
                        except Exception:
                            pass

                def _finalize() -> None:
                    q.put_nowait(json.dumps({"type": "status", "msg": "exit", "code": code}))
                    if latest is not None:
                        q.put_nowait(
                            json.dumps(
                                {
                                    "type": "final",
                                    "run_dir": str(latest),
                                    "sharpe_best": None if best is None else float(best),
                                }
                            )
                        )
                    if code != 0:
                        q.put_nowait(
                            json.dumps(
                                {
                                    "type": "error",
                                    "code": int(code),
                                    "last": last_line,
                                }
                            )
                        )
                    STATE.procs.pop(job_id, None)

                loop.call_soon_threadsafe(_finalize)

        await asyncio.to_thread(_stream_output)

    asyncio.create_task(_pump())
    return json_response({"job_id": job_id})
