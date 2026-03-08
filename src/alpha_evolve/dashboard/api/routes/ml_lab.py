from __future__ import annotations

import json
import time
from typing import Any

from django.http import HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from alpha_evolve.ml_lab.models import available_models_payload

from ..helpers import (
    ROOT,
    RE_PROGRESS,
    resolve_config_path,
    resolve_dataset_preset,
)
from ..http import json_error, json_response
from ..job_controller import get_dashboard_jobs
from ..ml_lab_runtime import (
    MLLabLaunchRequest,
    launch_ml_lab_job,
)
from ..ml_lab_runs import (
    find_ml_lab_runs,
    format_ml_lab_path_for_ui,
    plan_ml_lab_run,
    resolve_ml_lab_run_dir,
    safe_read_ml_lab_json,
)


JOB_STATE_RETENTION_SECONDS = 300.0


def list_models(_request: HttpRequest):
    return json_response({"models": available_models_payload()})


def list_runs(request: HttpRequest):
    limit_param = request.GET.get("limit", "50")
    try:
        limit = max(1, min(1000, int(limit_param)))
    except Exception:
        return json_error("limit must be an integer", 400)
    items: list[dict[str, Any]] = []
    for path in find_ml_lab_runs()[:limit]:
        summary = safe_read_ml_lab_json(path / "ml_summary.json") or {}
        meta = safe_read_ml_lab_json(path / "meta" / "run_metadata.json") or {}
        items.append(
            {
                "path": format_ml_lab_path_for_ui(path),
                "name": path.name,
                "status": meta.get("status"),
                "best_sharpe": summary.get("best_sharpe"),
                "completed": summary.get("completed"),
                "total": summary.get("total"),
                "started_at": meta.get("started_at"),
            }
        )
    return json_response(items)


def run_details(request: HttpRequest):
    run_dir = request.GET.get("run_dir")
    if not run_dir:
        return json_error("run_dir is required", 400)
    try:
        resolved = resolve_ml_lab_run_dir(run_dir)
    except ValueError as exc:
        return json_error(str(exc), 400)
    summary = safe_read_ml_lab_json(resolved / "ml_summary.json")
    results = safe_read_ml_lab_json(resolved / "ml_results.json")
    spec = safe_read_ml_lab_json(resolved / "ml_spec.json")
    meta = safe_read_ml_lab_json(resolved / "meta" / "run_metadata.json")
    return json_response(
        {
            "path": format_ml_lab_path_for_ui(resolved),
            "name": resolved.name,
            "summary": summary,
            "results": results,
            "spec": spec,
            "meta": meta,
        }
    )

@csrf_exempt
def start_run(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return json_error("Invalid JSON body", 400)
    if not isinstance(payload, dict):
        return json_error("Payload must be a JSON object", 400)

    dataset = str(payload.get("dataset") or "").strip().lower()
    cfg_path = payload.get("config")
    data_dir = payload.get("data_dir")

    if cfg_path:
        cfg_path = str(cfg_path)
        resolved_cfg = resolve_config_path(cfg_path)
        if resolved_cfg is None:
            return json_error(f"Config not found: {cfg_path}", 404)
        payload["config"] = str(resolved_cfg)
        cfg_path = str(resolved_cfg)
    elif dataset:
        preset = resolve_dataset_preset(dataset)
        if preset is None:
            return json_error(
                "Unknown dataset; use dataset=sp500, dataset=sp500_small, or provide a config path",
                400,
            )
        payload["config"] = str(preset)
    elif not data_dir:
        return json_error("config, dataset, or data_dir is required", 400)

    run_plan = plan_ml_lab_run(payload=payload, dataset=dataset, cfg_path=cfg_path)
    launch = launch_ml_lab_job(
        MLLabLaunchRequest(
            controller=get_dashboard_jobs(),
            root_dir=ROOT,
            run_dir=run_plan.run_dir,
            run_dir_label=run_plan.run_dir_label,
            spec_payload=payload,
            spec_path=run_plan.spec_path,
            progress_re=RE_PROGRESS,
            cleanup_delay_seconds=JOB_STATE_RETENTION_SECONDS,
        ),
    )

    return json_response({"job_id": launch.job_id, "run_dir": launch.run_dir_label})


@csrf_exempt
def stop_run(request: HttpRequest, job_id: str):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    controller = get_dashboard_jobs()
    stopped = controller.stop(job_id)
    if stopped:
        controller.touch_activity(
            job_id,
            last_message="Stop requested.",
            updated_at=time.time(),
        )
    return json_response({"stopped": bool(stopped)})
