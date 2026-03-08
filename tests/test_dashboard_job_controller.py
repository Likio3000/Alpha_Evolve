from __future__ import annotations

from pathlib import Path

from alpha_evolve.dashboard.api.job_controller import (
    DashboardJobController,
    get_dashboard_jobs,
    reset_dashboard_jobs,
    set_dashboard_jobs_for_tests,
)
from alpha_evolve.dashboard.api.jobs import JobHandle, JobState


def test_snapshot_activity_reads_file_tail_when_memory_log_empty(tmp_path: Path) -> None:
    state = JobState()
    controller = DashboardJobController(state)
    log_path = tmp_path / "job.log"
    log_path.write_text("line-1\nline-2\n", encoding="utf-8")
    state.init_activity("job-1", {"status": "running", "log_path": str(log_path)})

    payload = controller.snapshot_activity("job-1")

    assert payload["exists"] is True
    assert payload["running"] is False
    assert payload["log"].endswith("line-2\n")
    assert payload["log_path"] == str(log_path)


def test_summarize_counts_tracks_running_jobs() -> None:
    state = JobState()
    controller = DashboardJobController(state)
    state.handles["job-1"] = JobHandle()

    class RunningHandle:
        def is_running(self) -> bool:
            return True

    state.handles["job-2"] = RunningHandle()  # type: ignore[assignment]

    counts = controller.summarize_counts()

    assert counts == {"jobs_total": 2, "jobs_running": 1}


def test_dashboard_job_provider_can_be_swapped_for_tests() -> None:
    original = get_dashboard_jobs()
    replacement = DashboardJobController(JobState())
    try:
        set_dashboard_jobs_for_tests(replacement)
        assert get_dashboard_jobs() is replacement
    finally:
        reset_dashboard_jobs()
    assert get_dashboard_jobs() is original
