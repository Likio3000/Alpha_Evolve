from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..helpers import (
    ROOT,
    PIPELINE_DIR,
    read_best_sharpe_from_run,
    resolve_latest_run_dir,
)
from ..jobs import STATE


router = APIRouter()


def _labels_path() -> Path:
    return PIPELINE_DIR / ".run_labels.json"


def _load_labels() -> Dict[str, str]:
    p = _labels_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_labels(lbl: Dict[str, str]) -> None:
    p = _labels_path()
    try:
        p.write_text(json.dumps(lbl, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save labels")


def _find_runs() -> List[Path]:
    runs = [p for p in PIPELINE_DIR.glob("run_*") if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


@router.get("/api/runs")
def list_runs(limit: int = Query(default=50, ge=1, le=1000)) -> List[Dict[str, Any]]:
    labels = _load_labels()
    items: List[Dict[str, Any]] = []
    for p in _find_runs()[:limit]:
        rel = p.relative_to(ROOT)
        name = p.name
        label = labels.get(name)
        sharpe = read_best_sharpe_from_run(p)
        items.append({
            "path": str(rel),
            "name": name,
            "label": label,
            "sharpe_best": None if sharpe is None else float(sharpe),
        })
    return items


@router.get("/api/last-run")
def get_last_run() -> Dict[str, Any]:
    p = resolve_latest_run_dir()
    if p is None:
        return {"run_dir": None, "sharpe_best": None}
    sharpe = read_best_sharpe_from_run(p)
    return {"run_dir": str(p.relative_to(ROOT)), "sharpe_best": None if sharpe is None else float(sharpe)}


def _summary_csv(run_dir: Path) -> Optional[Path]:
    bt_dir = run_dir / "backtest_portfolio_csvs"
    if not bt_dir.exists():
        return None
    cand = sorted(bt_dir.glob("backtest_summary_top*.csv"))
    if not cand:
        return None
    return cand[-1]


@router.get("/api/backtest-summary")
def backtest_summary(run_dir: str) -> List[Dict[str, Any]]:
    p = (ROOT / run_dir).resolve()
    try:
        p.relative_to(PIPELINE_DIR.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="run_dir must be under pipeline_runs_cs/")
    csv_path = _summary_csv(p)
    if csv_path is None:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                # Best effort type normalization
                def _f(k: str) -> Optional[float]:
                    try:
                        return float(r.get(k, ""))
                    except Exception:
                        return None

                rows.append({
                    "AlphaID": r.get("AlphaID") or r.get("alpha_id"),
                    "TS": Path(r.get("TS") or r.get("TimeseriesFile") or "").name,
                    "TimeseriesFile": r.get("TimeseriesFile") or r.get("TS"),
                    "Sharpe": _f("Sharpe") or 0.0,
                    "AnnReturn": _f("AnnReturn") or 0.0,
                    "AnnVol": _f("AnnVol") or 0.0,
                    "MaxDD": _f("MaxDD") or 0.0,
                    "Turnover": _f("Turnover") or 0.0,
                    "Ops": r.get("Ops"),
                    "OriginalMetric": _f("OriginalMetric") or _f("original_metric") or _f("IC") or 0.0,
                    "Program": r.get("Program") or r.get("PROGRAM"),
                })
    except FileNotFoundError:
        return []
    return rows


def _resolve_ts_file(run_dir: Path, file: Optional[str], alpha_id: Optional[str]) -> Path:
    bt_dir = run_dir / "backtest_portfolio_csvs"
    if file:
        return bt_dir / Path(file).name
    # Need to map by alpha id via summary
    summary = _summary_csv(run_dir)
    if summary and alpha_id:
        try:
            with summary.open(newline="", encoding="utf-8") as fh:
                rdr = csv.DictReader(fh)
                for r in rdr:
                    aid = r.get("AlphaID") or r.get("alpha_id")
                    if str(aid) == str(alpha_id):
                        ts = r.get("TS") or r.get("TimeseriesFile")
                        if ts:
                            return bt_dir / Path(ts).name
        except Exception:
            pass
    # Fallback: not found
    raise HTTPException(status_code=404, detail="Timeseries not found")


@router.get("/api/alpha-timeseries")
def alpha_timeseries(run_dir: str, file: Optional[str] = None, alpha_id: Optional[str] = None) -> Dict[str, Any]:
    p = (ROOT / run_dir).resolve()
    try:
        p.relative_to(PIPELINE_DIR.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="run_dir must be under pipeline_runs_cs/")
    ts_path = _resolve_ts_file(p, file, alpha_id)
    if not ts_path.exists():
        raise HTTPException(status_code=404, detail="Timeseries CSV not found")
    dates: List[str] = []
    equity: List[float] = []
    ret_net: List[float] = []
    try:
        with ts_path.open(newline="", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                dates.append(str(r.get("date")))
                try:
                    equity.append(float(r.get("equity", "nan")))
                except Exception:
                    equity.append(float("nan"))
                try:
                    ret_net.append(float(r.get("ret_net", "nan")))
                except Exception:
                    ret_net.append(float("nan"))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read timeseries")
    return {"date": dates, "equity": equity, "ret_net": ret_net}


@router.get("/api/job-log/{job_id}")
def job_log(job_id: str) -> Dict[str, Any]:
    text = STATE.get_log_text(job_id)
    return {"log": text}


@router.get("/api/job-status/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    p = STATE.get_proc(job_id)
    if p is None:
        return {"exists": False, "running": False}
    try:
        running = p.poll() is None
    except Exception:
        running = False
    return {"exists": True, "running": bool(running)}


@router.post("/api/run-label")
def set_run_label(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = str(payload.get("path") or "").strip()
    label = str(payload.get("label") or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="Missing path")
    # Expect relative path under ROOT
    p = (ROOT / path).resolve()
    try:
        p.relative_to(PIPELINE_DIR.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="Path must be under pipeline_runs_cs/")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Run path not found")
    labels = _load_labels()
    if label:
        labels[p.name] = label
    else:
        labels.pop(p.name, None)
    _save_labels(labels)
    return {"ok": True}


@router.get("/api/run-asset")
def run_asset(run_dir: str, file: str, sub: Optional[str] = None) -> FileResponse:
    """Serve a file from a run directory safely (e.g., plots/topk_fitness_vs_ops.png).

    - run_dir: relative path under pipeline_runs_cs/
    - file: relative path within the run directory (e.g., 'plots/topk_fitness_vs_ops.png')
    - sub: optional subdirectory to join before file (ignored if file already has a directory component)
    """
    # Resolve and ensure run_dir under PIPELINE_DIR
    run_path = (ROOT / run_dir).resolve()
    try:
        run_path.relative_to(PIPELINE_DIR.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="run_dir must be under pipeline_runs_cs/")

    # Build candidate path
    fpath = Path(file)
    if fpath.is_absolute():
        raise HTTPException(status_code=400, detail="file must be relative")
    if sub and fpath.parent == Path('.'):
        fpath = Path(sub) / fpath
    full = (run_path / fpath).resolve()
    try:
        full.relative_to(run_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not full.exists():
        raise HTTPException(status_code=404, detail="File not found")
    # Infer media type
    media = "image/png" if full.suffix.lower() in (".png",) else None
    return FileResponse(str(full), media_type=media)
