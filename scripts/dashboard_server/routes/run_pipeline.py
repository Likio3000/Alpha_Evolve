from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from scripts.dashboard_server.jobs import STATE
from scripts.dashboard_server.helpers import (
    ROOT,
    build_pipeline_args,
    read_best_sharpe_from_run,
    resolve_latest_run_dir,
)


router = APIRouter()


@router.post("/api/pipeline/run")
async def start_pipeline_run(payload: Dict[str, Any]):
    job_id = STATE.__class__.__name__ + ":"  # keep a prefix, but uniqueness is from uuid in caller
    # Create queue and actual id here to align with legacy behavior
    import uuid as _uuid

    job_id = str(_uuid.uuid4())
    q = STATE.new_queue(job_id)

    args = build_pipeline_args(payload)

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
        re_sharpe = re.compile(r"Sharpe\\(best\\)\\s*=\\s*([+\\-]?[0-9.]+)")
        re_diag = re.compile(r"DIAG\\s+(\{.*\})$")
        re_progress = re.compile(r"PROGRESS\\s+(\{.*\})$")
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
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

    asyncio.create_task(_pump())
    return {"job_id": job_id}

