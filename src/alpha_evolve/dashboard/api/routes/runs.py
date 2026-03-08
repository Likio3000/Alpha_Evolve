from __future__ import annotations

import json
from django.http import HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from ..helpers import file_response
from ..http import json_error, json_response
from ..job_controller import get_dashboard_jobs
from ..pipeline_runs import (
    build_pending_pipeline_timeseries_payload,
    build_pipeline_backtest_summary_rows,
    build_job_log_payload,
    build_job_status_payload,
    build_last_pipeline_run_summary,
    build_pipeline_run_details,
    build_pipeline_timeseries_payload,
    build_pipeline_runs_listing,
    list_pipeline_run_assets,
    pipeline_backtest_summary_csv,
    resolve_pipeline_run_asset_path,
    resolve_pipeline_run_dir,
    resolve_pipeline_timeseries_file,
    update_pipeline_run_label,
)


def _summary_csv(run_dir):
    return pipeline_backtest_summary_csv(run_dir)


def list_runs(request: HttpRequest):
    limit_param = request.GET.get("limit", "50")
    try:
        limit = max(1, min(1000, int(limit_param)))
    except Exception:
        return json_error("limit must be an integer", 400)
    return json_response(build_pipeline_runs_listing(limit=limit))


def get_last_run(request: HttpRequest):
    return json_response(build_last_pipeline_run_summary())


def backtest_summary(request: HttpRequest):
    run_dir = request.GET.get("run_dir")
    if not run_dir:
        return json_error("run_dir is required", 400)
    try:
        p = resolve_pipeline_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)
    csv_path = _summary_csv(p)
    if csv_path is None:
        return json_response([])
    try:
        rows = build_pipeline_backtest_summary_rows(csv_path)
    except FileNotFoundError:
        return json_response([])
    return json_response(rows)


def alpha_timeseries(request: HttpRequest):
    run_dir = request.GET.get("run_dir")
    file = request.GET.get("file")
    alpha_id = request.GET.get("alpha_id")
    if not run_dir:
        return json_error("run_dir is required", 400)
    try:
        p = resolve_pipeline_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)
    summary_path = _summary_csv(p)
    summary_exists = summary_path is not None and summary_path.exists()
    if not summary_exists:
        return json_response(build_pending_pipeline_timeseries_payload(), status=202)

    try:
        ts_path = resolve_pipeline_timeseries_file(run_dir=p, file=file, alpha_id=alpha_id)
    except FileNotFoundError:
        return json_error("Timeseries CSV not found", 404)
    if not ts_path.exists():
        return json_error("Timeseries CSV not found", 404)
    try:
        payload = build_pipeline_timeseries_payload(ts_path)
    except Exception:
        return json_error("Failed to read timeseries", 500)
    return json_response(payload)


def job_log(request: HttpRequest, job_id: str):
    return json_response(build_job_log_payload(log_text=get_dashboard_jobs().get_log_text(job_id)))


def job_status(request: HttpRequest, job_id: str):
    return json_response(build_job_status_payload(handle=get_dashboard_jobs().get_handle(job_id)))


def job_activity(request: HttpRequest, job_id: str):
    return json_response(get_dashboard_jobs().snapshot_activity(job_id))


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
    try:
        response = update_pipeline_run_label(path=path, label=label)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except FileNotFoundError as exc:
        return json_error(str(exc), 404)
    except Exception:
        return json_error("Failed to save labels", 500)
    return json_response(response)


def run_asset(request: HttpRequest):
    run_dir = request.GET.get("run_dir")
    file = request.GET.get("file")
    sub = request.GET.get("sub")
    if not run_dir or not file:
        return json_error("run_dir and file are required", 400)
    try:
        run_path = resolve_pipeline_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)
    try:
        full = resolve_pipeline_run_asset_path(run_path=run_path, file=file, sub=sub)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except FileNotFoundError as exc:
        return json_error(str(exc), 404)
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
        resolved = resolve_pipeline_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)
    try:
        payload = list_pipeline_run_assets(resolved_run_dir=resolved, prefix=prefix, limit=limit)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception:
        return json_error("Failed to list artefacts", 500)
    return json_response(payload)


def run_details(request: HttpRequest):
    run_dir = request.GET.get("run_dir")
    if not run_dir:
        return json_error("run_dir is required", 400)
    try:
        resolved = resolve_pipeline_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)
    return json_response(build_pipeline_run_details(resolved_run_dir=resolved))
