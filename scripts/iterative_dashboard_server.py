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
from collections import deque
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
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

# Serve built UI (if available) under /ui
UI_DIR = ROOT / "dashboard-ui" / "dist"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


class JobState:
    def __init__(self) -> None:
        self.queues: Dict[str, asyncio.Queue] = {}
        self.procs: Dict[str, subprocess.Popen] = {}
        self.logs: Dict[str, deque[str]] = {}

    def new_queue(self, job_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self.queues[job_id] = q
        # Keep up to ~10k lines per job in memory
        self.logs[job_id] = deque(maxlen=10000)
        return q

    def get_queue(self, job_id: str) -> asyncio.Queue | None:
        return self.queues.get(job_id)

    def set_proc(self, job_id: str, proc: subprocess.Popen) -> None:
        self.procs[job_id] = proc

    def get_proc(self, job_id: str) -> subprocess.Popen | None:
        return self.procs.get(job_id)

    def add_log(self, job_id: str, line: str) -> None:
        if job_id not in self.logs:
            self.logs[job_id] = deque(maxlen=10000)
        self.logs[job_id].append(line)

    def get_log_text(self, job_id: str) -> str:
        buf = self.logs.get(job_id)
        if not buf:
            return ""
        return "\n".join(buf)

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

# Helper: import project-local config.py without colliding with external packages
def _load_project_config():
    """Load the project's config.py module by file path.

    Using a direct import by name ("import config") can accidentally import an
    unrelated third‑party package named "config". This loader guarantees we get
    the module that lives at ROOT/config.py.
    """
    import importlib.util
    cfg_path = ROOT / "config.py"
    spec = importlib.util.spec_from_file_location("ae_project_config", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config module from {cfg_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod
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

    # Build command (legacy: auto_improve-based runner)
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
        # Selection and exploration
        "selection_metric","ramp_fraction","ramp_min_gens","novelty_boost_w","novelty_struct_w","hof_corr_mode",
        "ic_tstat_w","temporal_decay_half_life","rank_softmax_beta_floor","rank_softmax_beta_target","corr_penalty_w",
        # Multi-objective elites
        "moea_enabled","moea_elite_frac",
        # Multi-fidelity
        "mf_enabled","mf_initial_fraction","mf_promote_fraction","mf_min_promote",
        # Cross-validation
        "cv_k_folds","cv_embargo",
        # Backtest ensemble
        "ensemble_mode","ensemble_size","ensemble_max_corr",
        # EvolutionParams / generation knobs
        "vector_ops_bias","relation_ops_weight","cs_ops_weight","default_op_weight",
        "ops_split_jitter","ops_split_base_setup","ops_split_base_predict","ops_split_base_update",
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
        # Local helper to read latest run dir without relying on globals
        def _latest() -> Path | None:
            latest_file = PIPELINE_DIR / "LATEST"
            try:
                if latest_file.exists():
                    p = latest_file.read_text().strip()
                    if p:
                        rp = Path(p)
                        return rp if rp.exists() else None
            except Exception:
                return None
            return None
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
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
                try:
                    print(line, flush=True)
                except Exception:
                    pass
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
            # Try to emit latest run dir and best Sharpe
            try:
                latest = _latest()
                if latest is not None:
                    bs = _read_best_sharpe_from_run(latest)
                    q.put_nowait(json.dumps({
                        "type": "final",
                        "run_dir": str(latest),
                        "sharpe_best": None if bs is None else float(bs),
                    }))
            except Exception:
                pass

    def _latest_from_marker() -> Path | None:
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


@app.post("/api/simple/run")
async def simple_run(payload: Dict[str, Any]):
    """Minimal runner for scripts: run_pipeline with dataset preset.

    Payload: {"dataset": "crypto"|"sp500", "generations": int, "data_dir"?: str}
    """
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


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline-first API (preferred)
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipeline_args(payload: Dict[str, Any]) -> list[str]:
    """Map JSON payload to a run_pipeline invocation.

    Recognized top-level keys:
      - generations (int)
      - dataset: "crypto" | "sp500" (sets default config if no explicit config)
      - config (path)
      - data_dir (path)
      - overrides: dict of CLI flags to values

    Any additional scalar keys are treated as CLI flags.
    """
    gens = int(payload.get("generations", 5))
    args: list[str] = ["uv", "run", "run_pipeline.py", str(gens)]
    dataset = str(payload.get("dataset", "")).strip().lower()
    cfg_path = payload.get("config")
    if not cfg_path and dataset:
        if dataset in ("crypto", "crypto_4h", "crypto4h"):
            cfg_path = str(ROOT / "configs" / "crypto_4h_fast.toml")
        elif dataset in ("sp500", "s&p500", "snp500"):
            cfg_path = str(ROOT / "configs" / "sp500.toml")
    if cfg_path:
        args += ["--config", str(cfg_path)]
    if payload.get("data_dir"):
        args += ["--data_dir", str(payload["data_dir"])]
    # Merge overrides
    overrides = dict(payload.get("overrides", {}))
    # If overrides include generations, use it as positional and remove from flags
    try:
        if "generations" in overrides:
            gens = int(overrides.pop("generations"))
    except Exception:
        pass
    # Guard: drop values that cannot be expressed on CLI or conflict with flags
    # - sector_mapping is a dict and not supported by the CLI helper
    overrides.pop("sector_mapping", None)
    # Also treat remaining simple keys as overrides (except reserved)
    reserved = {"generations", "dataset", "config", "data_dir", "overrides"}
    for k, v in payload.items():
        if k in reserved:
            continue
        if isinstance(v, (str, int, float, bool)):
            overrides[k] = v
    # Append overrides as CLI flags (use underscores: run_pipeline expects underscores)
    for k, v in overrides.items():
        # Only primitive scalar types are supported
        if not isinstance(v, (str, int, float, bool)):
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
            # if False, omit — only pass explicit negatives for known default-True flags
        else:
            args += [flag, str(v)]
    return args


@app.post("/api/pipeline/run")
async def start_pipeline_run(payload: Dict[str, Any]):
    """Start a single pipeline run (evolution + backtest) with provided config.

    Payload example:
      {"generations": 8, "dataset": "crypto", "data_dir": "./data",
       "overrides": {"moea_enabled": true, "ensemble_mode": true}}
    """
    job_id = str(uuid.uuid4())
    q = STATE.new_queue(job_id)

    args = _build_pipeline_args(payload)

    async def _pump():
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        STATE.set_proc(job_id, proc)
        q.put_nowait(json.dumps({"type": "status", "msg": "started", "args": args}))
        # Reuse the same regexes used above for DIAG/PROGRESS and best Sharpe lines
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
                # Buffer all lines for later retrieval via /api/job-log
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
            latest = _resolve_latest_run_dir()
            if latest is not None:
                bs = _read_best_sharpe_from_run(latest)
                q.put_nowait(json.dumps({
                    "type": "final",
                    "run_dir": str(latest),
                    "sharpe_best": None if bs is None else float(bs),
                }))

    asyncio.create_task(_pump())
    return {"job_id": job_id}


@app.get("/api/job-log/{job_id}")
async def job_log(job_id: str):
    """Return the captured terminal output for a job as text."""
    try:
        return {"log": STATE.get_log_text(job_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read job log: {e}")


# Config helpers
def _dataclass_defaults(dc_type) -> Dict[str, Any]:
    # Build a dict of field defaults from a dataclass type
    from dataclasses import fields as dc_fields, is_dataclass
    if not is_dataclass(dc_type):
        return {}
    try:
        inst = dc_type()  # type: ignore
        return {f.name: getattr(inst, f.name) for f in dc_fields(dc_type)}
    except Exception:
        d: Dict[str, Any] = {}
        for f in dc_fields(dc_type):
            if f.default is not None:
                d[f.name] = f.default
        return d


@app.get("/api/config/defaults")
async def config_defaults():
    choices = {
        "scale": ["zscore", "rank", "sign", "madz", "winsor"],
        "max_lookback_data_option": ["common_1200", "specific_long_10k", "full_overlap"],
        "selection_metric": ["ramped", "fixed", "ic", "auto", "phased"],
        "hof_corr_mode": ["flat", "per_bar"],
        "ensemble_weighting": ["equal", "risk_parity"],
        "split_weighting": ["equal", "by_points"],
    }
    try:
        cfg = _load_project_config()
        EvolutionConfig = getattr(cfg, "EvolutionConfig")
        BacktestConfig = getattr(cfg, "BacktestConfig")
        evo = _dataclass_defaults(EvolutionConfig)
        bt = _dataclass_defaults(BacktestConfig)
        if not isinstance(evo, dict) or not isinstance(bt, dict):
            raise RuntimeError("defaults not dict")
        return {"evolution": evo, "backtest": bt, "choices": choices}
    except Exception:
        # Fallback: return empty dicts with choices; UI will merge in preset values
        return {"evolution": {}, "backtest": {}, "choices": choices}


@app.get("/api/config/presets")
async def config_presets():
    presets = {
        "crypto": str(ROOT / "configs" / "crypto_4h_fast.toml"),
        "sp500": str(ROOT / "configs" / "sp500.toml"),
    }
    return {"presets": presets}


@app.get("/api/config/preset-values")
async def config_preset_values(dataset: str | None = None, path: str | None = None):
    """Return the evolution/backtest dicts from a preset TOML.

    - Pass `dataset=crypto|sp500` to use built-in presets
    - Or pass an explicit `path` to a TOML file
    """
    try:
        if path:
            toml_path = Path(path)
        else:
            ds = (dataset or "").strip().lower()
            if ds in ("crypto", "crypto_4h", "crypto4h"):
                toml_path = ROOT / "configs" / "crypto_4h_fast.toml"
            elif ds in ("sp500", "s&p500", "snp500"):
                toml_path = ROOT / "configs" / "sp500.toml"
            else:
                raise HTTPException(status_code=400, detail="Unknown dataset; provide ?dataset=crypto|sp500 or ?path=")
        if not toml_path.exists():
            raise HTTPException(status_code=404, detail=f"Preset not found: {toml_path}")
        # Prefer stdlib tomllib, fallback to tomli if needed
        try:
            import tomllib as _toml
        except Exception:
            import tomli as _toml  # type: ignore
        with open(toml_path, "rb") as fh:
            data = _toml.load(fh)
        # Merge TOML values over dataclass defaults so UI gets fully populated
        try:
            cfg = _load_project_config()
            EvolutionConfig = getattr(cfg, "EvolutionConfig")
            BacktestConfig = getattr(cfg, "BacktestConfig")
            evo_def = _dataclass_defaults(EvolutionConfig)
            bt_def = _dataclass_defaults(BacktestConfig)
        except Exception:
            evo_def = {}
            bt_def = {}
        evo = {**evo_def, **data.get("evolution", {})}
        bt = {**bt_def, **data.get("backtest", {})}
        choices = {
            "scale": ["zscore", "rank", "sign", "madz", "winsor"],
            "max_lookback_data_option": ["common_1200", "specific_long_10k", "full_overlap"],
            "selection_metric": ["ramped", "fixed", "ic", "auto", "phased"],
            "hof_corr_mode": ["flat", "per_bar"],
            "ensemble_weighting": ["equal", "risk_parity"],
            "split_weighting": ["equal", "by_points"],
        }
        return {"evolution": evo, "backtest": bt, "choices": choices, "path": str(toml_path)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load preset: {e}")


@app.get("/api/config/list")
async def config_list(limit: int = 200):
    """List local TOML files under configs/.

    Returns: {items: [{name, path, mtime}]}
    """
    try:
        items: list[dict[str, Any]] = []
        cfg_dir = ROOT / "configs"
        if cfg_dir.exists():
            for p in cfg_dir.glob("*.toml"):
                try:
                    st = p.stat()
                    items.append({
                        "name": p.name,
                        "path": str(p),
                        "mtime": st.st_mtime,
                    })
                except Exception:
                    continue
        items.sort(key=lambda x: x.get("mtime", 0), reverse=True)
        return {"items": items[: max(1, min(500, limit))]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list configs: {e}")


@app.post("/api/config/save")
async def config_save(payload: Dict[str, Any]):
    """Save evolution/backtest settings to configs/<name>.toml

    Expects JSON: {"name": "my_exp.toml", "evolution": {...}, "backtest": {...}}
    """
    try:
        name = str(payload.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not name.endswith(".toml"):
            name += ".toml"
        evo = payload.get("evolution", {}) or {}
        bt = payload.get("backtest", {}) or {}

        def _sc(v: Any) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return str(v)
            s = str(v)
            s = s.replace("\\", "\\\\").replace("\"", "\\\"")
            return f'"{s}"'

        def _sec(name: str, d: Dict[str, Any]) -> str:
            keys = sorted(d.keys())
            body = "\n".join(f"{k} = {_sc(d[k])}" for k in keys)
            return f"[{name}]\n{body}\n"

        txt = _sec("evolution", evo) + "\n" + _sec("backtest", bt)
        out_dir = ROOT / "configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / name
        out_path.write_text(txt)
        return {"saved": str(out_path)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")


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

# Backward-compat helper: some older paths referenced this name.
def _read_latest_run_dir() -> Path | None:  # noqa: N802 (compat)
    return _resolve_latest_run_dir()


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
        # Be lenient for live runs: return empty list instead of 404
        return []
    bt_dir = rd / "backtest_portfolio_csvs"
    if not bt_dir.exists():
        return []
    # Find the most recent summary JSON (fallback to CSV if JSON missing)
    cands_json = sorted(bt_dir.glob("backtest_summary_top*.json"))
    if cands_json:
        try:
            import json as _json
            with open(cands_json[-1]) as fh:
                return _json.load(fh)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load backtest summary JSON: {e}")
    # Fallback: CSV
    cands_csv = sorted(bt_dir.glob("backtest_summary_top*.csv"))
    if not cands_csv:
        # Not produced yet – return empty
        return []
    try:
        import csv
        rows = []
        with open(cands_csv[-1], newline="") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                # Coerce numeric fields to numbers for UI formatting
                def _f(key: str) -> float | None:
                    v = row.get(key)
                    if v is None or v == "":
                        return None
                    try:
                        return float(v)
                    except Exception:
                        return None
                def _i(key: str) -> int | None:
                    v = row.get(key)
                    if v is None or v == "":
                        return None
                    try:
                        return int(float(v))
                    except Exception:
                        return None
                out = dict(row)
                out["Sharpe"] = _f("Sharpe")
                out["AnnReturn"] = _f("AnnReturn")
                out["MaxDD"] = _f("MaxDD")
                out["Turnover"] = _f("Turnover")
                out["Ops"] = _i("Ops")
                rows.append(out)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load backtest summary CSV: {e}")


@app.get("/api/runs")
async def list_runs(limit: int = 20):
    """List recent pipeline runs with quick stats.

    Returns items sorted by mtime desc: {name, path, mtime, sharpe_best?}
    """
    try:
        items: list[dict[str, Any]] = []
        # Load user labels if present
        labels_path = PIPELINE_DIR / ".run_labels.json"
        labels: Dict[str, str] = {}
        try:
            if labels_path.exists():
                import json as _json
                with open(labels_path) as fh:
                    obj = _json.load(fh)
                if isinstance(obj, dict):
                    # keys are run directory basenames
                    labels = {str(k): str(v) for k, v in obj.items()}
        except Exception:
            labels = {}
        if PIPELINE_DIR.exists():
            for p in PIPELINE_DIR.iterdir():
                if not p.is_dir():
                    continue
                # Skip helper files
                if p.name.upper() in {"LATEST"}:
                    continue
                try:
                    stat = p.stat()
                    best = _read_best_sharpe_from_run(p)
                    items.append({
                        "name": p.name,
                        "path": str(p),
                        "mtime": stat.st_mtime,
                        "sharpe_best": best,
                        "label": labels.get(p.name),
                    })
                except Exception:
                    continue
        items.sort(key=lambda x: x.get("mtime", 0), reverse=True)
        return items[: max(1, min(200, limit))]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {e}")


def _safe_bt_dir_from_run(run_dir: Path | None) -> Path:
    rd = run_dir or _resolve_latest_run_dir()
    if rd is None:
        raise HTTPException(status_code=404, detail="No run dir found")
    bt_dir = rd / "backtest_portfolio_csvs"
    if not bt_dir.exists():
        raise HTTPException(status_code=404, detail="Backtest dir not found")
    return bt_dir


def _load_run_labels() -> Dict[str, str]:
    path = PIPELINE_DIR / ".run_labels.json"
    try:
        if not path.exists():
            return {}
        import json as _json
        with open(path) as fh:
            obj = _json.load(fh)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_run_labels(labels: Dict[str, str]) -> None:
    path = PIPELINE_DIR / ".run_labels.json"
    try:
        import json as _json
        PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            _json.dump(labels, fh, indent=2, sort_keys=True)
    except Exception:
        pass


@app.get("/api/run-labels")
async def get_run_labels():
    return _load_run_labels()


@app.post("/api/run-label")
async def set_run_label(payload: Dict[str, Any]):
    """Set a sticky human label for a run directory.

    Payload: {"path": "/abs/or/rel/path/to/run", "label": "My Experiment"}
    The label is stored under pipeline_runs_cs/.run_labels.json keyed by basename.
    """
    try:
        p = payload.get("path")
        label = str(payload.get("label", "")).strip()
        if not p or not label:
            raise HTTPException(status_code=400, detail="path and label required")
        run_path = Path(str(p))
        # Constrain to pipeline dir
        try:
            run_path = run_path if run_path.is_absolute() else (ROOT / Path(p))
            run_path = run_path.resolve()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid path")
        if PIPELINE_DIR not in run_path.parents:
            raise HTTPException(status_code=400, detail="path must be under pipeline_runs_cs")
        if not run_path.exists() or not run_path.is_dir():
            raise HTTPException(status_code=404, detail="run directory not found")
        labels = _load_run_labels()
        labels[run_path.name] = label
        _save_run_labels(labels)
        return {"ok": True, "name": run_path.name, "label": label}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set label: {e}")


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
    # Quieter server terminal: hide HTTP access logs so pipeline output stands out
    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)
@app.get("/ui-meta/evolution-params")
def get_evolution_params_ui_meta() -> Dict[str, Any]:
    """Expose UI metadata for EvolutionParams-related controls.

    The frontend can consume this to render labels, defaults and tooltips.
    """
    # Keep defaults aligned with config.EvolutionConfig and utils.EvolutionParams
    return {
        "schema_version": 1,
        "groups": [
            {
                "title": "Operator Biasing",
                "items": [
                    {"key": "vector_ops_bias", "label": "Vector Ops Bias", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Probability to force vector-output ops when sampling."},
                    {"key": "relation_ops_weight", "label": "Relation Ops Weight", "type": "float", "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5, "help": "Weight multiplier for relation_* ops during selection."},
                    {"key": "cs_ops_weight", "label": "Cross-sectional Ops Weight", "type": "float", "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.5, "help": "Weight multiplier for cs_* ops during selection."},
                    {"key": "default_op_weight", "label": "Default Op Weight", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "help": "Baseline weight for all ops."},
                ],
            },
            {
                "title": "Block Split",
                "items": [
                    {"key": "ops_split_base_setup", "label": "Setup Fraction", "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Base fraction of ops allocated to setup."},
                    {"key": "ops_split_base_predict", "label": "Predict Fraction", "type": "float", "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Base fraction of ops allocated to predict."},
                    {"key": "ops_split_base_update", "label": "Update Fraction", "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Base fraction of ops allocated to update."},
                    {"key": "ops_split_jitter", "label": "Split Jitter", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Randomness added to the block split when seeding fresh programs."},
                ],
            },
        ],
    }


@app.get("/ui-meta/pipeline-params")
def get_pipeline_params_ui_meta() -> Dict[str, Any]:
    """Expose metadata for common evolution/backtest pipeline parameters.

    The UI can render grouped controls and tooltips using this.
    Defaults mirror config.EvolutionConfig/BacktestConfig and CLI.
    """
    return {
        "schema_version": 1,
        "groups": [
            {
                "title": "Core Evolution",
                "items": [
                    {"key": "generations", "label": "Generations", "type": "int", "default": 5, "min": 1, "max": 1000, "step": 1, "help": "Number of evolutionary generations to run."},
                    {"key": "pop_size", "label": "Population Size", "type": "int", "default": 100, "min": 10, "max": 2000, "step": 10, "help": "Number of programs per generation."},
                    {"key": "tournament_k", "label": "Tournament K", "type": "int", "default": 10, "min": 2, "max": 200, "step": 1, "help": "Contestants per tournament when selecting parents."},
                    {"key": "elite_keep", "label": "Elites Kept", "type": "int", "default": 1, "min": 0, "max": 20, "step": 1, "help": "Top programs copied unchanged to next generation."},
                    {"key": "hof_size", "label": "HOF Size", "type": "int", "default": 20, "min": 1, "max": 200, "step": 1, "help": "Max Hall of Fame entries to keep."},
                    {"key": "hof_per_gen", "label": "HOF Per Gen", "type": "int", "default": 3, "min": 0, "max": 50, "step": 1, "help": "Top candidates added to HOF each generation (subject to filters)."},
                    {"key": "seed", "label": "Random Seed", "type": "int", "default": 42, "min": 0, "max": 2**31-1, "step": 1, "help": "Base seed for reproducibility."},
                    {"key": "fresh_rate", "label": "Fresh Rate", "type": "float", "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Fraction of population replaced with brand new programs each generation."},
                    {"key": "p_mut", "label": "Mutation Prob.", "type": "float", "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Probability to mutate a child after selection/crossover."},
                    {"key": "p_cross", "label": "Crossover Prob.", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Probability to crossover parents (0 uses cloning+mutation only)."},
                ],
            },
            {
                "title": "Selection & Ramping",
                "items": [
                    {"key": "selection_metric", "label": "Selection Metric", "type": "select", "default": "ramped", "choices": ["ramped", "fixed", "ic", "auto", "phased"], "help": "Parent selection criterion: ramped fitness, fixed final weights, pure IC, or mixed schedules."},
                    {"key": "ramp_fraction", "label": "Ramp Fraction", "type": "float", "default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Portion of total generations to gradually reach full penalty/weighting."},
                    {"key": "ramp_min_gens", "label": "Ramp Min Gens", "type": "int", "default": 5, "min": 0, "max": 100, "step": 1, "help": "Minimum generations for ramp to avoid premature exploitation."},
                    {"key": "ic_phase_gens", "label": "IC‑only Warmup Gens", "type": "int", "default": 0, "min": 0, "max": 100, "step": 1, "help": "Use pure IC for this many early generations (phased)."},
                    {"key": "rank_softmax_beta_floor", "label": "Rank Softmax Beta (Floor)", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "help": "Initial temperature for rank-based tournament sampling."},
                    {"key": "rank_softmax_beta_target", "label": "Rank Softmax Beta (Target)", "type": "float", "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1, "help": "Final temperature after ramp completes."},
                ],
            },
            {
                "title": "Novelty & Correlation",
                "items": [
                    {"key": "novelty_boost_w", "label": "Novelty Boost", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Bonus weight for low correlation vs Hall of Fame predictions."},
                    {"key": "novelty_struct_w", "label": "Structural Novelty", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Bonus for structural opcode-set dissimilarity vs HOF entries."},
                    {"key": "hof_corr_mode", "label": "HOF Corr Mode", "type": "select", "default": "flat", "choices": ["flat", "per_bar"], "help": "How to compute correlation vs HOF: flat (all points) or per-bar averaged."},
                    {"key": "corr_penalty_w", "label": "Correlation Penalty", "type": "float", "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Penalty weight for correlation vs HOF (applied in fitness)."},
                    {"key": "corr_cutoff", "label": "HOF Corr Cutoff", "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Drop candidates too correlated with HOF beyond this threshold."},
                ],
            },
            {
                "title": "Fitness Weights",
                "items": [
                    {"key": "sharpe_proxy_w", "label": "Sharpe Proxy Weight", "type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Weight for Sharpe-like proxy in fitness (0 uses IC-only)."},
                    {"key": "ic_std_penalty_w", "label": "IC Std Penalty", "type": "float", "default": 0.10, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Penalty weight for IC volatility."},
                    {"key": "turnover_penalty_w", "label": "Turnover Penalty", "type": "float", "default": 0.05, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Penalty for high position turnover (reduces trading)."},
                    {"key": "ic_tstat_w", "label": "IC t‑stat Weight", "type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "help": "Include IC t‑stat to reward stability in rank correlations."},
                    {"key": "temporal_decay_half_life", "label": "Temporal Decay Half‑life", "type": "float", "default": 0.0, "min": 0.0, "max": 10000.0, "step": 1.0, "help": "Exponential half-life in bars to weight recent data more (0 disables)."},
                ],
            },
            {
                "title": "Multi‑Objective & Fidelity",
                "items": [
                    {"key": "moea_enabled", "label": "Pareto Selection (MOEA)", "type": "bool", "default": False, "help": "Enable multi-objective selection (NSGA‑II‑like) for elites."},
                    {"key": "moea_elite_frac", "label": "Pareto Elite Fraction", "type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Portion of next generation chosen from the first Pareto front."},
                    {"key": "mf_enabled", "label": "Multi‑Fidelity Eval", "type": "bool", "default": False, "help": "Enable cheap first pass on truncated data then re‑evaluate top‑K fully."},
                    {"key": "mf_initial_fraction", "label": "MF Initial Fraction", "type": "float", "default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05, "help": "Fraction of bars used in the cheap first-pass evaluation."},
                    {"key": "mf_promote_fraction", "label": "MF Promote Fraction", "type": "float", "default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05, "help": "Fraction of population promoted to full evaluation."},
                    {"key": "mf_min_promote", "label": "MF Min Promote", "type": "int", "default": 8, "min": 1, "max": 200, "step": 1, "help": "Minimum number promoted regardless of fraction."},
                ],
            },
            {
                "title": "Cross‑Validation",
                "items": [
                    {"key": "cv_k_folds", "label": "CV K Folds", "type": "int", "default": 0, "min": 0, "max": 20, "step": 1, "help": "Use K>1 to enable CPCV‑style purged cross‑validation."},
                    {"key": "cv_embargo", "label": "CV Embargo (bars)", "type": "int", "default": 0, "min": 0, "max": 1000, "step": 1, "help": "Bars to embargo around each validation fold to reduce leakage."},
                ],
            },
            {
                "title": "Backtest Ensemble",
                "items": [
                    {"key": "ensemble_mode", "label": "Ensemble Backtest", "type": "bool", "default": False, "help": "Also backtest an ensemble of top alphas."},
                    {"key": "ensemble_size", "label": "Ensemble Size", "type": "int", "default": 0, "min": 0, "max": 100, "step": 1, "help": "Number of top alphas to include (0 disables)."},
                    {"key": "ensemble_max_corr", "label": "Ensemble Max Corr", "type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "help": "Target maximum pairwise correlation when selecting ensemble members."},
                ],
            },
            {
                "title": "Data & Run",
                "items": [
                    {"key": "data_dir", "label": "Data Directory", "type": "text", "default": "./data", "help": "Path to directory containing input CSV files."},
                    {"key": "bt_top", "label": "Backtest Top‑N", "type": "int", "default": 10, "min": 1, "max": 100, "step": 1, "help": "Number of evolved alphas to backtest and summarize."},
                    {"key": "no_clean", "label": "Keep Run Artefacts", "type": "bool", "default": False, "help": "Do not clean previous pipeline runs before starting."},
                    {"key": "dry_run", "label": "Dry Run", "type": "bool", "default": False, "help": "Print resolved configs and planned outputs, then exit."},
                    {"key": "out_summary", "label": "Write Summary JSON", "type": "bool", "default": True, "help": "Write SUMMARY.json with key artefacts for UI consumption."},
                ],
            },
        ],
    }
