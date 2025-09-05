from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from scripts.dashboard_server.jobs import STATE
from scripts.dashboard_server.helpers import ROOT, PIPELINE_DIR


router = APIRouter()


@router.post("/api/run")
async def start_run(payload: Dict[str, Any]):
    job_id = str(uuid.uuid4())
    q = STATE.new_queue(job_id)

    args = ["uv", "run", str(ROOT / "scripts" / "auto_improve.py")]
    for key in ("iters", "gens", "base_config", "data_dir", "bt_top", "no_clean", "dry_run", "sweep_capacity", "seeds", "out_summary"):
        if key not in payload:
            continue
        val = payload[key]
        flag = f"--{key.replace('_','-')}" if key in ("out_summary",) else f"--{key}"
        if isinstance(val, bool):
            if val:
                args.append(flag)
        else:
            args += [flag, str(val)]
    passthrough_keys = [
        "selection_metric",
        "ramp_fraction",
        "ramp_min_gens",
        "novelty_boost_w",
        "novelty_struct_w",
        "hof_corr_mode",
        "ic_tstat_w",
        "temporal_decay_half_life",
        "rank_softmax_beta_floor",
        "rank_softmax_beta_target",
        "corr_penalty_w",
        "moea_enabled",
        "moea_elite_frac",
        "mf_enabled",
        "mf_initial_fraction",
        "mf_promote_fraction",
        "mf_min_promote",
        "cv_k_folds",
        "cv_embargo",
        "ensemble_mode",
        "ensemble_size",
        "ensemble_max_corr",
        "vector_ops_bias",
        "relation_ops_weight",
        "cs_ops_weight",
        "default_op_weight",
        "ops_split_jitter",
        "ops_split_base_setup",
        "ops_split_base_predict",
        "ops_split_base_update",
    ]
    if any(k in payload for k in passthrough_keys):
        args.append("--")
        for key in passthrough_keys:
            if key not in payload:
                continue
            val = payload[key]
            flag = f"--{key}"
            if isinstance(val, bool):
                if val:
                    args.append(flag)
            else:
                args += [flag, str(val)]

    async def _pump():
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        STATE.set_proc(job_id, proc)
        q.put_nowait(json.dumps({"type": "status", "msg": "started", "args": args}))
        re_candidate = re.compile(r"^→ Candidate\s+(\d+)/(\d+):\s+(.*)$")
        re_sharpe = re.compile(r"Sharpe\(best\)\s*=\s*([+\-]?[0-9.]+)")
        re_diag = re.compile(r"DIAG\s+(\{.*\})$")
        re_progress = re.compile(r"PROGRESS\s+(\{.*\})$")
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                try:
                    print(line, flush=True)
                except Exception:
                    pass
                m = re_candidate.search(line)
                if m:
                    q.put_nowait(json.dumps({
                        "type": "candidate",
                        "idx": int(m.group(1)),
                        "total": int(m.group(2)),
                        "params": m.group(3),
                        "raw": line,
                    }))
                    continue
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
                    q.put_nowait(json.dumps({"type": "score", "sharpe_best": float(ms.group(1)), "raw": line}))
                else:
                    q.put_nowait(json.dumps({"type": "log", "raw": line}))
        finally:
            code = proc.wait()
            q.put_nowait(json.dumps({"type": "status", "msg": "exit", "code": code}))
            # attempt to fetch latest summary
            try:
                latest_file = PIPELINE_DIR / "LATEST"
                if latest_file.exists():
                    q.put_nowait(json.dumps({"type": "latest", "run_dir": latest_file.read_text().strip()}))
            except Exception:
                pass

    asyncio.create_task(_pump())
    return {"job_id": job_id}


@router.post("/api/simple/run")
async def simple_run(payload: Dict[str, Any]):
    job_id = str(uuid.uuid4())
    q = STATE.new_queue(job_id)
    gens = int(payload.get("generations", 8))
    ds = str(payload.get("dataset", "crypto")).strip().lower()
    if ds in ("crypto", "crypto_4h", "crypto4h"):
        cfg_path = str(ROOT / "configs" / "crypto_4h_fast.toml")
    elif ds in ("sp500", "s&p500", "snp500"):
        cfg_path = str(ROOT / "configs" / "sp500.toml")
    else:
        raise HTTPException(status_code=400, detail="dataset must be 'crypto' or 'sp500'")
    args: list[str] = ["uv", "run", "run_pipeline.py", str(gens), "--config", cfg_path]
    if payload.get("data_dir"):
        args += ["--data_dir", str(payload["data_dir"])]

    async def _pump():
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        STATE.set_proc(job_id, proc)
        q.put_nowait(json.dumps({"type": "status", "msg": "started", "args": args}))
        re_sharpe = re.compile(r"Sharpe\\(best\\)\\s*=\\s*([+\\-]?[0-9.]+)")
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                try:
                    print(line, flush=True)
                except Exception:
                    pass
                ms = re_sharpe.search(line)
                if ms:
                    q.put_nowait(json.dumps({"type": "score", "sharpe_best": float(ms.group(1)), "raw": line}))
                else:
                    q.put_nowait(json.dumps({"type": "log", "raw": line}))
        finally:
            code = proc.wait()
            q.put_nowait(json.dumps({"type": "status", "msg": "exit", "code": code}))

    asyncio.create_task(_pump())
    return {"job_id": job_id}


@router.get("/api/events/{job_id}")
async def sse_events(request: Request, job_id: str):
    q = STATE.get_queue(job_id)
    if q is None:
        raise HTTPException(status_code=404, detail="Unknown job id")

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.wait_for(q.get(), timeout=10.0)
                yield {"event": "message", "data": item}
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": json.dumps({"t": __import__("time").time()})}
                continue
    return EventSourceResponse(event_generator())


@router.post("/api/stop/{job_id}")
async def stop(job_id: str):
    ok = STATE.stop(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Unknown job id or already stopped")
    return {"stopped": True}

