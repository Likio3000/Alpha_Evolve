from __future__ import annotations

from pathlib import Path

from django.http import Http404, HttpResponse
from django.urls import path, re_path
from django.views.generic import RedirectView

from alpha_evolve.dashboard.api import health, metrics, ui_meta
from alpha_evolve.dashboard.api.routes import (
    config,
    run_pipeline,
    runs,
)
from alpha_evolve.dashboard.api.helpers import ROOT
from alpha_evolve.dashboard.api.http import json_response
from alpha_evolve.dashboard.api.helpers import file_response

UI_DIST = ROOT / "dashboard-ui" / "dist"


def _serve_ui(request, path: str = "index.html"):
    if not UI_DIST.exists():
        raise Http404("UI bundle not found")

    target = (path or "index.html").lstrip("/")
    full_path = (UI_DIST / target).resolve()
    try:
        full_path.relative_to(UI_DIST.resolve())
    except Exception:
        raise Http404("Invalid UI path")

    if full_path.exists() and full_path.is_file():
        return file_response(request.method, full_path, content_disposition="inline")

    accepts_html = "text/html" in request.headers.get("Accept", "")
    looks_like_asset = "." in Path(target).name
    if request.method in {"GET", "HEAD"} and accepts_html and not looks_like_asset:
        index_path = (UI_DIST / "index.html").resolve()
        if index_path.exists():
            return file_response(
                request.method, index_path, content_disposition="inline"
            )

    raise Http404()


def _serve_favicon(request):
    favicon = UI_DIST / "favicon.ico"
    if favicon.exists():
        return file_response(request.method, favicon, content_disposition="inline")
    return HttpResponse(status=204)


def _chrome_devtools_manifest(_request):
    # Chrome probes for this file to detect custom DevTools integrations.
    # Returning a minimal payload avoids noisy 404 warnings in the server logs.
    return json_response({"customFormatters": False})


urlpatterns = [
    path("", RedirectView.as_view(url="/ui/", permanent=False)),
    path("health", health.health),
    path("metrics", metrics.metrics),
    path("ui-meta/evolution-params", ui_meta.get_evolution_params_ui_meta),
    path("ui-meta/pipeline-params", ui_meta.get_pipeline_params_ui_meta),
    path("api/config/defaults", config.get_defaults),
    path("api/config/list", config.list_configs),
    path("api/config/presets", config.get_presets),
    path("api/config/preset-values", config.get_preset_values),
    path("api/config/save", config.save_config),
    path("api/pipeline/run", run_pipeline.start_pipeline_run),
    path("api/pipeline/stop/<str:job_id>", run_pipeline.stop),
    path("api/pipeline/events/<str:job_id>", run_pipeline.sse_events),
    path("api/runs", runs.list_runs),
    path("api/last-run", runs.get_last_run),
    path("api/backtest-summary", runs.backtest_summary),
    path("api/alpha-timeseries", runs.alpha_timeseries),
    path("api/job-log/<str:job_id>", runs.job_log),
    path("api/job-status/<str:job_id>", runs.job_status),
    path("api/job-activity/<str:job_id>", runs.job_activity),
    path("api/run-label", runs.set_run_label),
    path("api/run-details", runs.run_details),
    path("api/run-asset", runs.run_asset),
    path("api/run-assets", runs.run_assets),
    path("favicon.ico", _serve_favicon),
    path(".well-known/appspecific/com.chrome.devtools.json", _chrome_devtools_manifest),
]

urlpatterns.append(
    path("ui", RedirectView.as_view(url="/ui/", permanent=False, query_string=True))
)
urlpatterns.append(re_path(r"^ui/(?P<path>.*)$", _serve_ui))
