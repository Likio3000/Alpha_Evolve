from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pathlib import Path

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
from ..models import PipelineRunRequest


router = APIRouter()


@router.post("/api/pipeline/run")
async def start_pipeline_run(payload: PipelineRunRequest):
    # Early validation for clearer API errors
    pd = payload.model_dump()
    ds = (pd.get("dataset") or "").strip().lower()
    cfg_path = pd.get("config")
    if cfg_path:
        p = Path(str(cfg_path))
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Config not found: {p}")
    elif ds and ds not in ("crypto", "crypto_4h", "crypto4h", "sp500", "s&p500", "snp500"):
        raise HTTPException(status_code=400, detail="Unknown dataset; provide dataset=crypto|sp500 or a config path")
    job_id = STATE.__class__.__name__ + ":"  # keep a prefix, but uniqueness is from uuid in caller
    # Create queue and actual id here to align with legacy behavior
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
        re_sharpe = RE_SHARPE
        re_diag = RE_DIAG
        re_progress = RE_PROGRESS
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
                STATE.add_log(job_id, line)
                md = re_diag.search(line)
                if md:
                    try:
                        diag_obj = json.loads(md.group(1))
                        q.put_nowait(json.dumps({"type": "diag", "data": diag_obj}))
                    except Exception:
                        q.put_nowait(json.dumps({"type": "log", "raw": line}))
                    continue
                mp = re_progress.search(line)
                if mp:
                    try:
                        prog_obj = json.loads(mp.group(1))
                        q.put_nowait(json.dumps({"type": "progress", "data": prog_obj}))
                    except Exception:
                        q.put_nowait(json.dumps({"type": "log", "raw": line}))
                    continue
                ms = re_sharpe.search(line)
                if ms:
                    q.put_nowait(
                        json.dumps({"type": "score", "sharpe_best": float(ms.group(1)), "raw": line})
                    )
                else:
                    q.put_nowait(json.dumps({"type": "log", "raw": line}))
        finally:
            code = proc.wait()
            q.put_nowait(json.dumps({"type": "status", "msg": "exit", "code": code}))
            latest = resolve_latest_run_dir()
            context = STATE.pop_meta(job_id)
            if latest is not None:
                bs = read_best_sharpe_from_run(latest)
                q.put_nowait(
                    json.dumps(
                        {
                            "type": "final",
                            "run_dir": str(latest),
                            "sharpe_best": None if bs is None else float(bs),
                        }
                    )
                )
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
            if code != 0:
                q.put_nowait(json.dumps({"type": "error", "code": int(code), "last": last_line}))

    asyncio.create_task(_pump())
    return {"job_id": job_id}
