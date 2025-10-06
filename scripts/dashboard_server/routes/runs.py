from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.http import FileResponse, HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from ..helpers import (
    ROOT,
    PIPELINE_DIR,
    read_best_sharpe_from_run,
    resolve_latest_run_dir,
)
from ..http import json_error, json_response
from ..jobs import STATE


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
    p.write_text(json.dumps(lbl, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _find_runs() -> List[Path]:
    runs = [p for p in PIPELINE_DIR.glob("run_*") if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _format_path_for_ui(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        try:
            return str(p.relative_to(PIPELINE_DIR))
        except ValueError:
            return str(p)


def _resolve_run_dir(run_dir: str) -> Path:
    candidate = Path(run_dir)
    base_candidates = []
    if candidate.is_absolute():
        base_candidates.append(candidate.resolve())
    else:
        base_candidates.append((PIPELINE_DIR / candidate).resolve())
        base_candidates.append((ROOT / candidate).resolve())
    for resolved in base_candidates:
        try:
            resolved.relative_to(PIPELINE_DIR.resolve())
        except ValueError:
            continue
        if resolved.exists():
            return resolved
    raise ValueError("run_dir must resolve under pipeline_runs_cs/")


def list_runs(request: HttpRequest):
    limit_param = request.GET.get("limit", "50")
    try:
        limit = max(1, min(1000, int(limit_param)))
    except Exception:
        return json_error("limit must be an integer", 400)
    labels = _load_labels()
    items: List[Dict[str, Any]] = []
    for p in _find_runs()[:limit]:
        rel = _format_path_for_ui(p)
        name = p.name
        label = labels.get(name)
        sharpe = read_best_sharpe_from_run(p)
        items.append({
            "path": rel,
            "name": name,
            "label": label,
            "sharpe_best": None if sharpe is None else float(sharpe),
        })
    return json_response(items)


def get_last_run(request: HttpRequest):
    p = resolve_latest_run_dir()
    if p is None:
        return json_response({"run_dir": None, "sharpe_best": None})
    sharpe = read_best_sharpe_from_run(p)
    return json_response({"run_dir": _format_path_for_ui(p), "sharpe_best": None if sharpe is None else float(sharpe)})


def _summary_csv(run_dir: Path) -> Optional[Path]:
    bt_dir = run_dir / "backtest_portfolio_csvs"
    if not bt_dir.exists():
        return None
    cand = sorted(bt_dir.glob("backtest_summary_top*.csv"))
    if not cand:
        return None
    return cand[-1]


def backtest_summary(request: HttpRequest):
    run_dir = request.GET.get("run_dir")
    if not run_dir:
        return json_error("run_dir is required", 400)
    try:
        p = _resolve_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)
    csv_path = _summary_csv(p)
    if csv_path is None:
        return json_response([])
    rows: List[Dict[str, Any]] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
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
        return json_response([])
    return json_response(rows)


def _resolve_ts_file(run_dir: Path, file: Optional[str], alpha_id: Optional[str]) -> Path:
    bt_dir = run_dir / "backtest_portfolio_csvs"
    if file:
        return bt_dir / Path(file).name
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
    raise FileNotFoundError("Timeseries not found")


def alpha_timeseries(request: HttpRequest):
    run_dir = request.GET.get("run_dir")
    file = request.GET.get("file")
    alpha_id = request.GET.get("alpha_id")
    if not run_dir:
        return json_error("run_dir is required", 400)
    try:
        p = _resolve_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)
    try:
        ts_path = _resolve_ts_file(p, file, alpha_id)
    except FileNotFoundError:
        return json_error("Timeseries CSV not found", 404)
    if not ts_path.exists():
        return json_error("Timeseries CSV not found", 404)
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
        return json_error("Failed to read timeseries", 500)
    return json_response({"date": dates, "equity": equity, "ret_net": ret_net})


def job_log(request: HttpRequest, job_id: str):
    text = STATE.get_log_text(job_id)
    return json_response({"log": text})


def job_status(request: HttpRequest, job_id: str):
    handle = STATE.get_handle(job_id)
    if handle is None:
        return json_response({"exists": False, "running": False})
    running = handle.is_running()
    return json_response({"exists": True, "running": bool(running)})


@csrf_exempt
def set_run_label(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return json_error("Invalid JSON body", 400)
    path = str(payload.get("path") or "").strip()
    label = str(payload.get("label") or "").strip()
    if not path:
        return json_error("Missing path", 400)
    p = (ROOT / path).resolve()
    try:
        p.relative_to(PIPELINE_DIR.resolve())
    except Exception:
        return json_error("Path must be under pipeline_runs_cs/", 400)
    if not p.exists():
        return json_error("Run path not found", 404)
    labels = _load_labels()
    if label:
        labels[p.name] = label
    else:
        labels.pop(p.name, None)
    try:
        _save_labels(labels)
    except Exception:
        return json_error("Failed to save labels", 500)
    return json_response({"ok": True})


def run_asset(request: HttpRequest):
    run_dir = request.GET.get("run_dir")
    file = request.GET.get("file")
    sub = request.GET.get("sub")
    if not run_dir or not file:
        return json_error("run_dir and file are required", 400)
    run_path = (ROOT / run_dir).resolve()
    try:
        run_path.relative_to(PIPELINE_DIR.resolve())
    except Exception:
        return json_error("run_dir must be under pipeline_runs_cs/", 400)

    fpath = Path(file)
    if fpath.is_absolute():
        return json_error("file must be relative", 400)
    if sub and fpath.parent == Path('.'):
        fpath = Path(sub) / fpath
    full = (run_path / fpath).resolve()
    try:
        full.relative_to(run_path)
    except Exception:
        return json_error("Invalid file path", 400)
    if not full.exists():
        return json_error("File not found", 404)
    media = "image/png" if full.suffix.lower() in (".png",) else None
    response = FileResponse(open(full, "rb"), content_type=media)
    response["Content-Disposition"] = f'inline; filename="{full.name}"'
    return response
