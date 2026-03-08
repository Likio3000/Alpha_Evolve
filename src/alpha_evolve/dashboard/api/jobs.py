from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import subprocess
import threading
from queue import Queue
import asyncio
import time
import uuid


@dataclass
class JobHandle:
    """Track how a background job can be monitored or stopped."""

    proc: Optional[Any] = None
    task: Optional[asyncio.Task] = None
    stop_cb: Optional[Callable[[], None]] = None

    def is_running(self) -> bool:
        if self.proc is not None:
            try:
                if hasattr(self.proc, "poll"):
                    return self.proc.poll() is None  # subprocess.Popen API
                if hasattr(self.proc, "is_alive"):
                    return self.proc.is_alive()  # multiprocessing.Process API
            except Exception:
                return False
        if self.task is not None:
            return not self.task.done()
        return False

    def stop(self) -> bool:
        try:
            if self.stop_cb is not None:
                self.stop_cb()
                return True
            if self.proc is not None:
                if hasattr(self.proc, "terminate"):
                    self.proc.terminate()
                try:
                    if hasattr(self.proc, "wait"):
                        self.proc.wait(timeout=5)
                    elif hasattr(self.proc, "join"):
                        self.proc.join(timeout=5)
                except Exception:
                    if hasattr(self.proc, "kill"):
                        self.proc.kill()
                return True
            if self.task is not None:
                self.task.cancel()
                return True
        except Exception:
            return False
        return False


class JobState:
    """Lightweight process + log manager for the dashboard server."""

    def __init__(self) -> None:
        self.queues: Dict[str, Queue] = {}
        self.handles: Dict[str, JobHandle] = {}
        self.logs: Dict[str, deque[str]] = {}
        self.meta: Dict[str, Any] = {}
        self.activity: Dict[str, Dict[str, Any]] = {}
        self.cleanup_timers: Dict[str, threading.Timer] = {}

    def new_queue(self, job_id: str):
        self._cancel_cleanup(job_id)
        q: Queue[str] = Queue()
        self.queues[job_id] = q
        # Keep up to ~10k lines per job in memory
        self.logs[job_id] = deque(maxlen=10000)
        return q

    def initialize_job(
        self,
        *,
        job_id: str | None = None,
        meta: Any = None,
        activity: Dict[str, Any] | None = None,
    ) -> tuple[str, Queue]:
        resolved_job_id = job_id or str(uuid.uuid4())
        queue = self.new_queue(resolved_job_id)
        if meta is not None:
            self.set_meta(resolved_job_id, meta)
        self.init_activity(resolved_job_id, activity)
        return resolved_job_id, queue

    def get_queue(self, job_id: str):
        return self.queues.get(job_id)

    def set_proc(self, job_id: str, proc: subprocess.Popen) -> None:
        self.handles[job_id] = JobHandle(proc=proc)

    def set_task(self, job_id: str, task: asyncio.Task, stop_cb: Callable[[], None] | None = None) -> None:
        self.handles[job_id] = JobHandle(task=task, stop_cb=stop_cb)

    def set_handle(self, job_id: str, handle: JobHandle) -> None:
        self.handles[job_id] = handle

    def get_proc(self, job_id: str) -> subprocess.Popen | None:
        handle = self.handles.get(job_id)
        return handle.proc if handle else None

    def get_handle(self, job_id: str) -> JobHandle | None:
        return self.handles.get(job_id)

    def add_log(self, job_id: str, line: str) -> None:
        if job_id not in self.logs:
            self.logs[job_id] = deque(maxlen=10000)
        self.logs[job_id].append(line)

    def get_log_text(self, job_id: str) -> str:
        buf = self.logs.get(job_id)
        if not buf:
            return ""
        return "\n".join(buf)

    def set_meta(self, job_id: str, payload: Any) -> None:
        self.meta[job_id] = payload

    def pop_meta(self, job_id: str) -> Any:
        return self.meta.pop(job_id, None)

    def init_activity(self, job_id: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        data = payload.copy() if payload else {}
        self.activity[job_id] = data
        return data

    def get_activity(self, job_id: str) -> Dict[str, Any] | None:
        return self.activity.get(job_id)

    def update_activity(self, job_id: str, **updates: Any) -> Dict[str, Any]:
        data = self.activity.setdefault(job_id, {})
        data.update(updates)
        return data

    def touch_activity(self, job_id: str, **updates: Any) -> Dict[str, Any]:
        updates.setdefault("updated_at", time.time())
        return self.update_activity(job_id, **updates)

    def set_log_path(self, job_id: str, log_path: str) -> Dict[str, Any]:
        return self.touch_activity(job_id, log_path=log_path)

    def append_activity_summary(self, job_id: str, summary: Any, limit: int = 400) -> None:
        data = self.activity.setdefault(job_id, {})
        history = data.setdefault("summaries", [])
        if isinstance(history, list):
            history.append(summary)
            if len(history) > limit:
                del history[0 : len(history) - limit]
        else:
            data["summaries"] = [summary]

    def append_meta_sequence(
        self,
        job_id: str,
        key: str,
        item: Any,
        *,
        limit: int,
    ) -> list[Any]:
        meta = self.meta.setdefault(job_id, {})
        if not isinstance(meta, dict):
            meta = {}
            self.meta[job_id] = meta
        sequence = meta.setdefault(key, [])
        if not isinstance(sequence, list):
            sequence = []
            meta[key] = sequence
        sequence.append(item)
        if len(sequence) > limit:
            del sequence[0 : len(sequence) - limit]
        return sequence

    def clear_activity(self, job_id: str) -> None:
        self.activity.pop(job_id, None)

    def stop(self, job_id: str) -> bool:
        handle = self.handles.get(job_id)
        if handle is None:
            return False
        return handle.stop()

    def clear_handle(self, job_id: str) -> None:
        self.handles.pop(job_id, None)

    def clear_job(self, job_id: str) -> None:
        self._cancel_cleanup(job_id)
        self.handles.pop(job_id, None)
        self.queues.pop(job_id, None)
        self.logs.pop(job_id, None)
        self.meta.pop(job_id, None)
        self.activity.pop(job_id, None)

    def schedule_cleanup(self, job_id: str, delay_seconds: float = 300.0) -> None:
        self._cancel_cleanup(job_id)
        if delay_seconds <= 0:
            self.clear_job(job_id)
            return

        def _cleanup() -> None:
            self.clear_job(job_id)

        timer = threading.Timer(delay_seconds, _cleanup)
        timer.daemon = True
        self.cleanup_timers[job_id] = timer
        timer.start()

    def _cancel_cleanup(self, job_id: str) -> None:
        timer = self.cleanup_timers.pop(job_id, None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass


# Global job state singleton for ease of wiring across routers
STATE = JobState()
