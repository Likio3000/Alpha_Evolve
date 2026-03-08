from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any

from .jobs import JobHandle, JobState, STATE


def _read_log_tail(path: Path, max_lines: int = 2000) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = deque(fh, maxlen=max_lines if max_lines > 0 else None)
        return "".join(lines)
    except Exception:
        return None


@dataclass(frozen=True)
class DashboardJobActivityPayload:
    exists: bool
    running: bool
    log: str
    status: Any | None = None
    last_message: Any | None = None
    sharpe_best: Any | None = None
    progress: Any | None = None
    updated_at: Any | None = None
    run_dir: Any | None = None
    summaries: list[Any] | None = None
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "exists": self.exists,
            "running": self.running,
            "log": self.log,
        }
        for key in (
            "status",
            "last_message",
            "sharpe_best",
            "progress",
            "updated_at",
            "run_dir",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.summaries is not None:
            payload["summaries"] = self.summaries
        if self.log_path is not None:
            payload["log_path"] = self.log_path
        return payload


class DashboardJobController:
    def __init__(self, state: JobState) -> None:
        self.state = state

    def initialize_job(self, **kwargs: Any) -> tuple[str, Queue]:
        return self.state.initialize_job(**kwargs)

    def set_handle(self, job_id: str, handle: JobHandle) -> None:
        self.state.set_handle(job_id, handle)

    def set_proc(self, job_id: str, proc: Any) -> None:
        self.state.set_proc(job_id, proc)

    def clear_job(self, job_id: str) -> None:
        self.state.clear_job(job_id)

    def clear_handle(self, job_id: str) -> None:
        self.state.clear_handle(job_id)

    def schedule_cleanup(self, job_id: str, delay_seconds: float) -> None:
        self.state.schedule_cleanup(job_id, delay_seconds=delay_seconds)

    def get_queue(self, job_id: str) -> Queue | None:
        return self.state.get_queue(job_id)

    def stop(self, job_id: str) -> bool:
        return self.state.stop(job_id)

    def touch_activity(self, job_id: str, **updates: Any) -> dict[str, Any]:
        return self.state.touch_activity(job_id, **updates)

    def set_log_path(self, job_id: str, log_path: str) -> dict[str, Any]:
        return self.state.set_log_path(job_id, log_path)

    def get_activity(self, job_id: str) -> dict[str, Any] | None:
        return self.state.get_activity(job_id)

    def pop_meta(self, job_id: str) -> Any:
        return self.state.pop_meta(job_id)

    def get_log_text(self, job_id: str) -> str:
        return self.state.get_log_text(job_id)

    def get_handle(self, job_id: str) -> JobHandle | None:
        return self.state.get_handle(job_id)

    def add_log(self, job_id: str, line: str) -> None:
        self.state.add_log(job_id, line)

    def append_activity_summary(self, job_id: str, summary: Any, limit: int = 400) -> None:
        self.state.append_activity_summary(job_id, summary, limit=limit)

    def append_meta_sequence(
        self,
        job_id: str,
        key: str,
        item: Any,
        *,
        limit: int,
    ) -> list[Any]:
        return self.state.append_meta_sequence(job_id, key, item, limit=limit)

    def summarize_counts(self) -> dict[str, int]:
        jobs_total = len(self.state.handles)
        jobs_running = sum(1 for handle in self.state.handles.values() if handle.is_running())
        return {
            "jobs_total": jobs_total,
            "jobs_running": jobs_running,
        }

    def snapshot_activity(self, job_id: str) -> dict[str, Any]:
        activity = self.state.get_activity(job_id)
        handle = self.state.get_handle(job_id)
        running = bool(handle and handle.is_running())
        log_text = self.state.get_log_text(job_id)
        exists = activity is not None or handle is not None
        kwargs: dict[str, Any] = {
            "exists": bool(exists),
            "running": running,
        }
        if isinstance(activity, dict):
            for key in (
                "status",
                "last_message",
                "sharpe_best",
                "progress",
                "updated_at",
                "run_dir",
            ):
                if key in activity:
                    kwargs[key] = activity[key]
            summaries = activity.get("summaries")
            if isinstance(summaries, list):
                kwargs["summaries"] = summaries
            log_path = activity.get("log_path")
            if isinstance(log_path, str):
                kwargs["log_path"] = log_path
                if (not log_text or not log_text.strip()) and log_path:
                    tail = _read_log_tail(Path(log_path))
                    if tail is not None:
                        log_text = tail
        return DashboardJobActivityPayload(log=log_text or "", **kwargs).to_dict()


_DEFAULT_DASHBOARD_JOBS = DashboardJobController(STATE)
_dashboard_jobs = _DEFAULT_DASHBOARD_JOBS


def get_dashboard_jobs() -> DashboardJobController:
    return _dashboard_jobs


def set_dashboard_jobs_for_tests(controller: DashboardJobController) -> None:
    global _dashboard_jobs
    _dashboard_jobs = controller


def reset_dashboard_jobs() -> None:
    global _dashboard_jobs
    _dashboard_jobs = _DEFAULT_DASHBOARD_JOBS


DASHBOARD_JOBS = _DEFAULT_DASHBOARD_JOBS
