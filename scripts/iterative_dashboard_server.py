#!/usr/bin/env python3
"""
iterative_dashboard_server.py – Lightweight backend for the Iterative UI

Provides endpoints to:
- Launch scripts/auto_improve.py with JSON params.
- Stream live stdout as Server‑Sent Events (SSE) for progress updates.
- Query latest pipeline run summary and best Sharpe so far.

Run:
  uv run scripts/iterative_dashboard_server.py

Then open the UI and switch to "Dashboard" mode (http://localhost:5173 by default).
"""
from __future__ import annotations
import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline_runs_cs"

app = FastAPI(title="Alpha Evolve Iterative Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobState:
    def __init__(self) -> None:
        self.queues: Dict[str, asyncio.Queue] = {}
        self.procs: Dict[str, subprocess.Popen] = {}

    def new_queue(self, job_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self.queues[job_id] = q
        return q

    def get_queue(self, job_id: str) -> asyncio.Queue | None:
        return self.queues.get(job_id)

    def set_proc(self, job_id: str, proc: subprocess.Popen) -> None:
        self.procs[job_id] = proc

    def get_proc(self, job_id: str) -> subprocess.Popen | None:
        return self.procs.get(job_id)

    def stop(self, job_id: str) -> bool:
        p = self.procs.get(job_id)
        if p is None:
            return False
        try:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
            return True
        except Exception:
            return False


STATE = JobState()


def _read_best_sharpe_from_run(run_dir: Path) -> float | None:
    bt_dir = run_dir / "backtest_portfolio_csvs"
    candidates = sorted(bt_dir.glob("backtest_summary_top*.csv"))
    if not candidates:
        return None
    csv_path = candidates[-1]
    try:
        import csv
        best = None
        with open(csv_path, newline="") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                try:
                    s = float(row.get("Sharpe", "nan"))
                except Exception:
                    continue
                if best is None or s > best:
                    best = s
        return best
    except Exception:
        return None


@app.post("/api/run")
async def start_run(payload: Dict[str, Any]):
    """Start auto_improve with given parameters; returns a job_id.

    Payload JSON is forwarded as CLI flags, e.g. {"iters": 2, "gens": 10}.
    """
    job_id = str(uuid.uuid4())
    q = STATE.new_queue(job_id)

    # Build command
    args = [
        "uv", "run", str(ROOT / "scripts" / "auto_improve.py")
    ]
    # core flags first
    for key in ("iters", "gens", "base_config", "data_dir", "bt_top", "no_clean", "dry_run",
                "sweep_capacity", "seeds", "out_summary"):
        if key not in payload:
            continue
        val = payload[key]
        flag = f"--{key.replace('_','-')}" if key in ("out_summary",) else f"--{key}"  # keep simple
        if isinstance(val, bool):
            if val:
                args.append(flag)
        else:
            args += [flag, str(val)]
    # passthrough flags appended after separator
    passthrough_keys = [
        "selection_metric","ramp_fraction","ramp_min_gens","novelty_boost_w","novelty_struct_w","hof_corr_mode",
        "ic_tstat_w","temporal_decay_half_life","rank_softmax_beta_floor","rank_softmax_beta_target","corr_penalty_w"
    ]
    has_pt = any(k in payload for k in passthrough_keys)
    if has_pt:
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
        proc = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        STATE.set_proc(job_id, proc)
        q.put_nowait(json.dumps({"type": "status", "msg": "started", "args": args}))
        # Regexes to extract useful markers
        re_candidate = re.compile(r"^→ Candidate\s+(\d+)/(\d+):\s+(.*)$")
        re_sharpe = re.compile(r"Sharpe\(best\)\s*=\s*([+\-]?[0-9.]+)")
        re_diag = re.compile(r"DIAG\s+(\{.*\})$")
        re_progress = re.compile(r"PROGRESS\s+(\{.*\})$")
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                # Parse a few structured events
                m = re_candidate.search(line)
                if m:
                    q.put_nowait(json.dumps({
                        "type": "candidate",
                        "idx": int(m.group(1)),
                        "total": int(m.group(2)),
                        "params": m.group(3),
                        "raw": line,
                    }))
                else:
                    md = re_diag.search(line)
                    if md:
                        try:
                            diag_obj = json.loads(md.group(1))
                            q.put_nowait(json.dumps({"type": "diag", "data": diag_obj}))
                        except Exception:
                            q.put_nowait(json.dumps({"type": "log", "raw": line}))
                    else:
                        mp = re_progress.search(line)
                        if mp:
                            try:
                                prog_obj = json.loads(mp.group(1))
                                q.put_nowait(json.dumps({"type": "progress", "data": prog_obj}))
                            except Exception:
                                q.put_nowait(json.dumps({"type": "log", "raw": line}))
                        else:
                            ms = re_sharpe.search(line)
                            if ms:
                                q.put_nowait(json.dumps({
                                    "type": "score",
                                    "sharpe_best": float(ms.group(1)),
                                    "raw": line,
                                }))
                            else:
                                q.put_nowait(json.dumps({"type": "log", "raw": line}))
        finally:
            code = proc.wait()
            q.put_nowait(json.dumps({"type": "status", "msg": "exit", "code": code}))
            # Try to emit latest run dir and best sharpe
            latest = _read_latest_run_dir()
            if latest is not None:
                bs = _read_best_sharpe_from_run(latest)
                q.put_nowait(json.dumps({
                    "type": "final",
                    "run_dir": str(latest),
                    "sharpe_best": None if bs is None else float(bs),
                }))

    def _read_latest_run_dir() -> Path | None:
        latest_file = PIPELINE_DIR / "LATEST"
        try:
            if latest_file.exists():
                p = latest_file.read_text().strip()
                if p:
                    run_path = Path(p)
                    return run_path if run_path.exists() else None
        except Exception:
            pass
        return None

    # Spawn pumping task in background
    asyncio.create_task(_pump())
    return {"job_id": job_id}


@app.get("/api/events/{job_id}")
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
                yield {
                    "event": "message",
                    "data": item,
                }
            except asyncio.TimeoutError:
                # Keep-alive
                yield {"event": "ping", "data": json.dumps({"t": time.time()})}
                continue
    return EventSourceResponse(event_generator())


@app.get("/api/last-run")
async def last_run():
    latest = PIPELINE_DIR / "LATEST"
    if not latest.exists():
        return {"run_dir": None, "sharpe_best": None}
    p = latest.read_text().strip()
    if not p:
        return {"run_dir": None, "sharpe_best": None}
    run_dir = Path(p)
    bs = _read_best_sharpe_from_run(run_dir)
    return {"run_dir": str(run_dir), "sharpe_best": bs}


@app.post("/api/stop/{job_id}")
async def stop(job_id: str):
    ok = STATE.stop(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Unknown job id or already stopped")
    return {"stopped": True}


def _resolve_latest_run_dir() -> Path | None:
    latest = PIPELINE_DIR / "LATEST"
    try:
        if latest.exists():
            p = latest.read_text().strip()
            if p:
                run_path = Path(p)
                return run_path if run_path.exists() else None
    except Exception:
        return None
    return None


@app.get("/api/diagnostics")
async def diagnostics(run_dir: str | None = None):
    """Return diagnostics.json content for a run (latest if not specified)."""
    rd = Path(run_dir) if run_dir else _resolve_latest_run_dir()
    if rd is None:
        raise HTTPException(status_code=404, detail="No run dir found")
    diag_path = rd / "diagnostics.json"
    if not diag_path.exists():
        raise HTTPException(status_code=404, detail="diagnostics.json not found")
    try:
        import json as _json
        with open(diag_path) as fh:
            return _json.load(fh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load diagnostics: {e}")


@app.get("/api/backtest-summary")
async def backtest_summary(run_dir: str | None = None):
    """Return the backtest summary JSON for a run (latest if not specified)."""
    rd = Path(run_dir) if run_dir else _resolve_latest_run_dir()
    if rd is None:
        raise HTTPException(status_code=404, detail="No run dir found")
    bt_dir = rd / "backtest_portfolio_csvs"
    if not bt_dir.exists():
        raise HTTPException(status_code=404, detail="Backtest dir not found")
    # Find the most recent summary JSON
    cands = sorted(bt_dir.glob("backtest_summary_top*.json"))
    if not cands:
        raise HTTPException(status_code=404, detail="Backtest summary JSON not found")
    try:
        import json as _json
        with open(cands[-1]) as fh:
            return _json.load(fh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load backtest summary: {e}")


def _safe_bt_dir_from_run(run_dir: Path | None) -> Path:
    rd = run_dir or _resolve_latest_run_dir()
    if rd is None:
        raise HTTPException(status_code=404, detail="No run dir found")
    bt_dir = rd / "backtest_portfolio_csvs"
    if not bt_dir.exists():
        raise HTTPException(status_code=404, detail="Backtest dir not found")
    return bt_dir


@app.get("/api/alpha-timeseries")
async def alpha_timeseries(run_dir: str | None = None, alpha_id: str | None = None, file: str | None = None):
    """Return per-alpha timeseries JSON for plotting.

    Accepts either an `alpha_id` like "Alpha_01" or a `file` (basename) such as
    "alpha_01_timeseries.csv". The lookup is constrained to the run's
    backtest_portfolio_csvs directory for safety.
    """
    try:
        rd = Path(run_dir) if run_dir else _resolve_latest_run_dir()
        bt_dir = _safe_bt_dir_from_run(rd)
        target: Path | None = None
        if file:
            # Constrain to basename under bt_dir
            target = bt_dir / Path(file).name
        elif alpha_id:
            name = alpha_id.strip().lower().replace("alpha_", "alpha_")
            # Ensure two-digit formatting if user passes Alpha_1
            try:
                suffix = alpha_id.split("_")[-1]
                n = int(suffix)
                name = f"alpha_{n:02d}_timeseries.csv"
            except Exception:
                name = f"{alpha_id}_timeseries.csv"
            target = bt_dir / name
        else:
            raise HTTPException(status_code=400, detail="alpha_id or file is required")
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"Timeseries not found: {target.name}")
        # Parse CSV into light JSON
        import csv
        dates: list[str] = []
        equity: list[float] = []
        drawdown: list[float] = []
        exposure: list[float] = []
        stop_hits: list[float] = []
        ret_net: list[float] = []
        with open(target, newline="") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                dates.append(str(row.get("date", "")))
                def _f(key: str) -> float:
                    try:
                        return float(row.get(key, "nan"))
                    except Exception:
                        return float("nan")
                ret_net.append(_f("ret_net"))
                equity.append(_f("equity"))
                exposure.append(_f("exposure_mult"))
                drawdown.append(_f("drawdown"))
                try:
                    stop_hits.append(float(row.get("stop_hits", 0) or 0))
                except Exception:
                    stop_hits.append(0.0)
        return {
            "file": target.name,
            "date": dates,
            "ret_net": ret_net,
            "equity": equity,
            "exposure_mult": exposure,
            "drawdown": drawdown,
            "stop_hits": stop_hits,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load timeseries: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
