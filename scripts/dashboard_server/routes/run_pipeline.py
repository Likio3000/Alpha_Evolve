from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pathlib import Path

from scripts.dashboard_server.jobs import STATE
from scripts.dashboard_server.helpers import (
    ROOT,
    build_pipeline_args,
    read_best_sharpe_from_run,
    resolve_latest_run_dir,
    RE_SHARPE,
    RE_DIAG,
    RE_PROGRESS,
)
from scripts.dashboard_server.models import PipelineRunRequest


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

    async def _pump():
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
            if code != 0:
                q.put_nowait(json.dumps({"type": "error", "code": int(code), "last": last_line}))

    asyncio.create_task(_pump())
    return {"job_id": job_id}
