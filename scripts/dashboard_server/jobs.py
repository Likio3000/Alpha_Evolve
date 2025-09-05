from __future__ import annotations

from collections import deque
from typing import Dict
import subprocess


class JobState:
    """Lightweight process + log manager for the dashboard server."""

    def __init__(self) -> None:
        self.queues: Dict[str, object] = {}
        self.procs: Dict[str, subprocess.Popen] = {}
        self.logs: Dict[str, deque[str]] = {}

    def new_queue(self, job_id: str):
        import asyncio

        q: asyncio.Queue = asyncio.Queue()
        self.queues[job_id] = q
        # Keep up to ~10k lines per job in memory
        self.logs[job_id] = deque(maxlen=10000)
        return q

    def get_queue(self, job_id: str):
        return self.queues.get(job_id)

    def set_proc(self, job_id: str, proc: subprocess.Popen) -> None:
        self.procs[job_id] = proc

    def get_proc(self, job_id: str) -> subprocess.Popen | None:
        return self.procs.get(job_id)

    def add_log(self, job_id: str, line: str) -> None:
        if job_id not in self.logs:
            self.logs[job_id] = deque(maxlen=10000)
        self.logs[job_id].append(line)

    def get_log_text(self, job_id: str) -> str:
        buf = self.logs.get(job_id)
        if not buf:
            return ""
        return "\n".join(buf)

    def stop(self, job_id: str) -> bool:
        p = self.procs.get(job_id)
        if p is None:
            return False
        try:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
            return True
        except Exception:
            return False

