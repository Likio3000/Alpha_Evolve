from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.http import HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from ..helpers import (
    ROOT,
    PIPELINE_DIR,
    file_response,
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


def _safe_read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _read_codename(run_dir: Path) -> Optional[str]:
    meta = _safe_read_json(run_dir / "meta" / "run_metadata.json")
    if isinstance(meta, dict):
        codename = meta.get("codename")
        if isinstance(codename, str):
            codename = codename.strip()
            if codename:
                return codename
    return None


def _resolve_run_dir(run_dir: str) -> Path:
    candidate = Path(run_dir)
    base_candidates = []
    pipeline_root = PIPELINE_DIR.resolve()
    if candidate.is_absolute():
        base_candidates.append(candidate.resolve())
    else:
        stripped = candidate
        if stripped.parts and stripped.parts[0] == pipeline_root.name:
            stripped = (
                Path(*stripped.parts[1:]) if len(stripped.parts) > 1 else Path(".")
            )
        base_candidates.append((pipeline_root / stripped).resolve())
        base_candidates.append((PIPELINE_DIR.parent / candidate).resolve())
        base_candidates.append((ROOT / candidate).resolve())
    for resolved in base_candidates:
        try:
            resolved.relative_to(pipeline_root)
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
        dir_name = p.name
        display_name = _read_codename(p) or dir_name
        label = labels.get(dir_name)
        sharpe = read_best_sharpe_from_run(p)
        items.append(
            {
                "path": rel,
                "name": display_name,
                "label": label,
                "sharpe_best": None if sharpe is None else float(sharpe),
            }
        )
    return json_response(items)


def get_last_run(request: HttpRequest):
    p = resolve_latest_run_dir()
    if p is None:
        return json_response({"run_dir": None, "sharpe_best": None})
    sharpe = read_best_sharpe_from_run(p)
    return json_response(
        {
            "run_dir": _format_path_for_ui(p),
            "sharpe_best": None if sharpe is None else float(sharpe),
        }
    )


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

                rows.append(
                    {
                        "AlphaID": r.get("AlphaID") or r.get("alpha_id"),
                        "TS": Path(r.get("TS") or r.get("TimeseriesFile") or "").name,
                        "TimeseriesFile": r.get("TimeseriesFile") or r.get("TS"),
                        "Sharpe": _f("Sharpe") or 0.0,
                        "AnnReturn": _f("AnnReturn") or 0.0,
                        "AnnVol": _f("AnnVol") or 0.0,
                        "MaxDD": _f("MaxDD") or 0.0,
                        "Turnover": _f("Turnover") or 0.0,
                        "Ops": r.get("Ops"),
                        "OriginalMetric": _f("OriginalMetric")
                        or _f("original_metric")
                        or _f("IC")
                        or 0.0,
                        "Program": r.get("Program") or r.get("PROGRAM"),
                    }
                )
    except FileNotFoundError:
        return json_response([])
    return json_response(rows)


def _resolve_ts_file(
    run_dir: Path, file: Optional[str], alpha_id: Optional[str]
) -> Path:
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
    summary_path = _summary_csv(p)
    summary_exists = summary_path is not None and summary_path.exists()

    try:
        ts_path = _resolve_ts_file(p, file, alpha_id)
    except FileNotFoundError:
        if not summary_exists:
            return json_response(
                {"date": [], "equity": [], "ret_net": [], "pending": True}, status=202
            )
        return json_error("Timeseries CSV not found", 404)
    if not ts_path.exists():
        if not summary_exists:
            return json_response(
                {"date": [], "equity": [], "ret_net": [], "pending": True}, status=202
            )
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


def _read_log_tail(path: Path, max_lines: int = 2000) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = deque(fh, maxlen=max_lines if max_lines > 0 else None)
        return "".join(lines)
    except Exception:
        return None


def job_activity(request: HttpRequest, job_id: str):
    activity = STATE.get_activity(job_id)
    handle = STATE.get_handle(job_id)
    running = bool(handle and handle.is_running())
    log_text = STATE.get_log_text(job_id)
    exists = activity is not None or handle is not None
    payload: Dict[str, Any] = {
        "exists": bool(exists),
        "running": running,
    }
    if isinstance(activity, dict):
        for key in ("status", "last_message", "sharpe_best", "progress", "updated_at", "run_dir"):
            if key in activity:
                payload[key] = activity[key]
        summaries = activity.get("summaries")
        if isinstance(summaries, list):
            payload["summaries"] = summaries
        log_path = activity.get("log_path")
        if isinstance(log_path, str):
            payload["log_path"] = log_path
            if (not log_text or not log_text.strip()) and log_path:
                tail = _read_log_tail(Path(log_path))
                if tail is not None:
                    log_text = tail
    if not log_text:
        log_text = ""
    payload["log"] = log_text
    return json_response(payload)


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
    try:
        run_path = _resolve_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)

    fpath = Path(file)
    if fpath.is_absolute():
        return json_error("file must be relative", 400)
    if sub and fpath.parent == Path("."):
        fpath = Path(sub) / fpath
    full = (run_path / fpath).resolve()
    try:
        full.relative_to(run_path)
    except Exception:
        return json_error("Invalid file path", 400)
    if not full.exists():
        return json_error("File not found", 404)
    return file_response(
        request.method, full, content_disposition="inline", filename=full.name
    )


def run_assets(request: HttpRequest):
    """List previewable artefacts under a run directory.

    This powers the dashboard artefact browser; returned paths are relative to the run root.
    """

    run_dir = request.GET.get("run_dir")
    prefix = request.GET.get("prefix", "")
    limit_param = request.GET.get("limit", "500")
    if not run_dir:
        return json_error("run_dir is required", 400)
    try:
        limit = max(1, min(5000, int(limit_param)))
    except Exception:
        return json_error("limit must be an integer", 400)
    try:
        resolved = _resolve_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)

    base = resolved
    if prefix:
        candidate = (resolved / prefix).resolve()
        try:
            candidate.relative_to(resolved)
        except Exception:
            return json_error("prefix must stay within run_dir", 400)
        base = candidate

    if not base.exists():
        return json_response({"items": []})

    allowed = {".csv", ".json", ".jsonl", ".log", ".md", ".png", ".txt"}
    items: List[str] = []
    try:
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed:
                continue
            try:
                rel = path.relative_to(resolved)
            except ValueError:
                continue
            items.append(str(rel))
            if len(items) >= limit:
                break
    except Exception:
        return json_error("Failed to list artefacts", 500)

    items.sort()
    return json_response({"items": items})


def run_details(request: HttpRequest):
    run_dir = request.GET.get("run_dir")
    if not run_dir:
        return json_error("run_dir is required", 400)
    try:
        resolved = _resolve_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)

    sharpe = read_best_sharpe_from_run(resolved)
    labels = _load_labels()
    payload: Dict[str, Any] = {
        "path": _format_path_for_ui(resolved),
        "name": _read_codename(resolved) or resolved.name,
        "label": labels.get(resolved.name),
        "sharpe_best": None if sharpe is None else float(sharpe),
    }

    summary = _safe_read_json(resolved / "SUMMARY.json")
    if summary is not None:
        payload["summary"] = summary

    meta_dir = resolved / "meta"
    if meta_dir.exists():
        ui_context = _safe_read_json(meta_dir / "ui_context.json")
        if ui_context is not None:
            payload["ui_context"] = ui_context

        extra_meta: Dict[str, Any] = {}
        for key, filename in (
            ("evolution_config", "evolution_config.json"),
            ("backtest_config", "backtest_config.json"),
            ("run_metadata", "run_metadata.json"),
            ("data_alignment", "data_alignment.json"),
        ):
            data = _safe_read_json(meta_dir / filename)
            if data is not None:
                extra_meta[key] = data
        if extra_meta:
            payload["meta"] = extra_meta

    return json_response(payload)
