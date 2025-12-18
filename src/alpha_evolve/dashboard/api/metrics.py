from __future__ import annotations

import os
from pathlib import Path

from django.http import HttpRequest, HttpResponse

from alpha_evolve.experiments import ExperimentRegistry

from .helpers import PIPELINE_DIR, ROOT
from .jobs import STATE


def _resolve_experiments_db() -> Path:
    override = os.environ.get("AE_EXPERIMENTS_DB")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate
    return (ROOT / "artifacts" / "experiments" / "experiments.sqlite").resolve()


def _count_sessions(db_path: Path) -> tuple[int, int]:
    if not db_path.exists():
        return 0, 0
    reg = ExperimentRegistry(db_path)
    with reg.connect() as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        active = int(
            conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE status IN ('running', 'awaiting_approval')"
            ).fetchone()[0]
        )
    return total, active


def metrics(_request: HttpRequest):
    jobs_total = len(STATE.handles)
    jobs_running = sum(1 for handle in STATE.handles.values() if handle.is_running())
    runs_total = len([p for p in PIPELINE_DIR.glob("run_*") if p.is_dir()])

    sess_total = 0
    sess_active = 0
    try:
        sess_total, sess_active = _count_sessions(_resolve_experiments_db())
    except Exception:
        sess_total, sess_active = 0, 0

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
    lines.append("# HELP alpha_evolve_experiment_sessions_total Total number of experiment sessions recorded.")
    lines.append("# TYPE alpha_evolve_experiment_sessions_total gauge")
    lines.append(f"alpha_evolve_experiment_sessions_total {sess_total}")
    lines.append("# HELP alpha_evolve_experiment_sessions_active Number of active experiment sessions.")
    lines.append("# TYPE alpha_evolve_experiment_sessions_active gauge")
    lines.append(f"alpha_evolve_experiment_sessions_active {sess_active}")
    lines.append("")

    return HttpResponse("\n".join(lines), content_type="text/plain; version=0.0.4; charset=utf-8")

