from __future__ import annotations

from pathlib import Path

from django.http import Http404
from django.urls import path, re_path
from django.views.generic import RedirectView
from django.views.static import serve as static_serve

from scripts.dashboard_server import health, ui_meta
from scripts.dashboard_server.routes import (
    config,
    run_pipeline,
    runs,
)
from scripts.dashboard_server.helpers import ROOT

UI_DIST = ROOT / "dashboard-ui" / "dist"


def _serve_ui(request, path: str = "index.html"):
    if not UI_DIST.exists():
        raise Http404("UI bundle not found")

    target = (path or "index.html").lstrip("/")

    try:
        return static_serve(request, target, document_root=str(UI_DIST))
    except Http404:
        accepts_html = "text/html" in request.headers.get("Accept", "")
        looks_like_asset = "." in Path(target).name
        if request.method in {"GET", "HEAD"} and accepts_html and not looks_like_asset:
            return static_serve(request, "index.html", document_root=str(UI_DIST))
        raise


urlpatterns = [
    path("", RedirectView.as_view(url="/ui/", permanent=False)),
    path("health", health.health),
    path("ui-meta/evolution-params", ui_meta.get_evolution_params_ui_meta),
    path("ui-meta/pipeline-params", ui_meta.get_pipeline_params_ui_meta),
    path("api/config/defaults", config.get_defaults),
    path("api/config/list", config.list_configs),
    path("api/config/presets", config.get_presets),
    path("api/config/preset-values", config.get_preset_values),
    path("api/config/save", config.save_config),
    path("api/pipeline/run", run_pipeline.start_pipeline_run),
    path("api/runs", runs.list_runs),
    path("api/last-run", runs.get_last_run),
    path("api/backtest-summary", runs.backtest_summary),
    path("api/alpha-timeseries", runs.alpha_timeseries),
    path("api/job-log/<str:job_id>", runs.job_log),
    path("api/job-status/<str:job_id>", runs.job_status),
    path("api/run-label", runs.set_run_label),
    path("api/run-asset", runs.run_asset),
]

if UI_DIST.exists():
    urlpatterns.append(re_path(r"^ui/?$", _serve_ui))
    urlpatterns.append(re_path(r"^ui/(?P<path>.*)$", _serve_ui))
