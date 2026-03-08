from __future__ import annotations

import os
from pathlib import Path

from django.http import HttpRequest, HttpResponse

from .job_controller import get_dashboard_jobs
from .helpers import PIPELINE_DIR, ROOT


def metrics(_request: HttpRequest):
    job_counts = get_dashboard_jobs().summarize_counts()
    jobs_total = job_counts["jobs_total"]
    jobs_running = job_counts["jobs_running"]
    runs_total = len([p for p in PIPELINE_DIR.glob("run_*") if p.is_dir()])

    lines: list[str] = []
    lines.append("# HELP alpha_evolve_dashboard_jobs_total Total number of tracked dashboard jobs.")
    lines.append("# TYPE alpha_evolve_dashboard_jobs_total gauge")
    lines.append(f"alpha_evolve_dashboard_jobs_total {jobs_total}")
    lines.append("# HELP alpha_evolve_dashboard_jobs_running Number of dashboard jobs currently running.")
    lines.append("# TYPE alpha_evolve_dashboard_jobs_running gauge")
    lines.append(f"alpha_evolve_dashboard_jobs_running {jobs_running}")
    lines.append("# HELP alpha_evolve_pipeline_runs_total Total number of pipeline run directories.")
    lines.append("# TYPE alpha_evolve_pipeline_runs_total gauge")
    lines.append(f"alpha_evolve_pipeline_runs_total {runs_total}")
    lines.append("")

    return HttpResponse("\n".join(lines), content_type="text/plain; version=0.0.4; charset=utf-8")
