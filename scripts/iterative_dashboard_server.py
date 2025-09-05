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
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from scripts.dashboard_server.jobs import JobState, STATE as _GLOBAL_STATE
from scripts.dashboard_server.ui_meta import router as ui_meta_router
from scripts.dashboard_server.routes.run_auto_improve import router as auto_router
from scripts.dashboard_server.routes.run_pipeline import router as pipeline_router
from scripts.dashboard_server.helpers import read_best_sharpe_from_run as _read_best_sharpe_from_run
from scripts.dashboard_server.helpers import resolve_latest_run_dir as _resolve_latest_run_dir

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
app.include_router(ui_meta_router)
app.include_router(auto_router)
app.include_router(pipeline_router)
app.include_router(ui_meta_router)




STATE = _GLOBAL_STATE


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


# moved to scripts.dashboard_server.routes.run_auto_improve


# moved to scripts.dashboard_server.routes.run_auto_improve


# moved to scripts.dashboard_server.routes.run_auto_improve


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


# moved to scripts.dashboard_server.routes.run_pipeline


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
