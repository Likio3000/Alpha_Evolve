from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from queue import Queue
from typing import Any

from .job_controller import DashboardJobController


GEN_HISTORY_LIMIT = 2000


def _append_generation_history(
    controller: DashboardJobController,
    job_id: str,
    summary: dict[str, Any],
) -> None:
    controller.append_meta_sequence(
        job_id,
        "gen_history",
        summary,
        limit=GEN_HISTORY_LIMIT,
    )


def _write_run_metadata(
    *,
    controller: DashboardJobController,
    job_id: str,
    run_dir: str,
) -> None:
    context = controller.pop_meta(job_id)
    history = None
    if isinstance(context, dict):
        history = context.pop("gen_history", None)
    if not context:
        return
    try:
        run_path = Path(run_dir).resolve()
        if not run_path.exists():
            return
        meta_dir = run_path / "meta"
        meta_dir.mkdir(exist_ok=True)
        context_out = dict(context)
        context_out["run_dir"] = str(run_path)
        with open(meta_dir / "ui_context.json", "w", encoding="utf-8") as fh:
            json.dump(context_out, fh, indent=2)
        if history:
            with open(meta_dir / "gen_summary.jsonl", "w", encoding="utf-8") as fh_hist:
                for entry in history:
                    fh_hist.write(json.dumps(entry))
                    fh_hist.write("\n")
    except Exception:
        pass


class PipelineEventForwarder:
    def __init__(
        self,
        *,
        controller: DashboardJobController,
        job_id: str,
        event_queue: Any,
        client_queue: Queue,
        cleanup_delay_seconds: float,
    ) -> None:
        self._controller = controller
        self._job_id = job_id
        self._event_queue = event_queue
        self._client_queue = client_queue
        self._cleanup_delay_seconds = cleanup_delay_seconds
        self._log_handle = None

    def _log_line(self, line: str) -> None:
        if not isinstance(line, str):
            return
        activity = self._controller.get_activity(self._job_id) or {}
        log_path = activity.get("log_path")
        if not isinstance(log_path, str):
            return
        try:
            if self._log_handle is None:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                self._log_handle = open(log_path, "a", encoding="utf-8")
            if line.endswith("\n"):
                self._log_handle.write(line)
            else:
                self._log_handle.write(line + "\n")
            self._log_handle.flush()
        except Exception:
            pass

    def _touch_activity(self, **updates: Any) -> None:
        self._controller.touch_activity(self._job_id, updated_at=time.time(), **updates)

    def _handle_progress_summary(self, data: dict[str, Any]) -> None:
        _append_generation_history(self._controller, self._job_id, data)
        self._controller.append_activity_summary(self._job_id, data)
        self._touch_activity(progress=data)

    def _handle_status_exit(self, item: dict[str, Any]) -> None:
        try:
            code = int(item.get("code", 1))
        except Exception:
            code = 1
        success = code == 0
        current = self._controller.get_activity(self._job_id) or {}
        current_message = current.get("last_message")
        generic_messages = {
            None,
            "",
            "Pipeline started.",
            "Pipeline error.",
            "Pipeline stopped.",
            "Stop requested.",
            "Stop requested…",
        }
        final_message = (
            "Pipeline finished."
            if success
            else (
                current_message
                if current_message not in generic_messages
                else "Pipeline stopped."
            )
        )
        self._touch_activity(
            status="complete" if success else "error",
            last_message=final_message,
        )
        if not success:
            self._controller.pop_meta(self._job_id)

    def _handle_final(self, item: dict[str, Any]) -> None:
        run_dir = item.get("run_dir")
        if isinstance(run_dir, str) and run_dir.strip():
            _write_run_metadata(controller=self._controller, job_id=self._job_id, run_dir=run_dir)
            self._touch_activity(run_dir=run_dir.strip())
        sharpe = item.get("sharpe_best")
        try:
            value = float(sharpe)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            self._touch_activity(sharpe_best=value)
        self._touch_activity(status="complete")

    def _handle_item(self, item: dict[str, Any]) -> bool:
        event_type = item.get("type")
        raw_line = item.get("raw")
        if event_type == "__complete__":
            return False
        self._touch_activity()
        if event_type == "log":
            if isinstance(raw_line, str):
                text = raw_line.strip()
                if text:
                    self._touch_activity(last_message=text)
        elif event_type == "progress":
            data = item.get("data")
            subtype = item.get("subtype") or (data.get("type") if isinstance(data, dict) else None)
            if subtype == "gen_progress" and isinstance(data, dict):
                self._touch_activity(progress=data)
            elif subtype == "gen_summary" and isinstance(data, dict):
                self._handle_progress_summary(data)
        elif event_type == "gen_summary":
            data = item.get("data")
            if isinstance(data, dict):
                self._handle_progress_summary(data)
        elif event_type == "score":
            sharpe = item.get("sharpe_best")
            try:
                value = float(sharpe)
            except (TypeError, ValueError):
                value = None
            if value is not None:
                self._touch_activity(sharpe_best=value)
        elif event_type == "status":
            msg = item.get("msg")
            if msg == "exit":
                self._handle_status_exit(item)
            elif isinstance(msg, str):
                mapped = "Pipeline started." if msg == "started" else msg
                self._touch_activity(status="running", last_message=mapped)
        elif event_type == "error":
            detail = item.get("detail")
            if isinstance(detail, str) and detail.strip():
                message = detail
                self._log_line(detail)
            else:
                message = "Pipeline error."
            self._touch_activity(status="error", last_message=message)
        elif event_type == "final":
            self._handle_final(item)

        if isinstance(raw_line, str):
            self._controller.add_log(self._job_id, raw_line)
            if raw_line:
                self._log_line(raw_line)
        try:
            self._client_queue.put_nowait(json.dumps(item))
        except Exception:
            pass
        return True

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                item = await loop.run_in_executor(None, self._event_queue.get)
                if not isinstance(item, dict):
                    break
                if not self._handle_item(item):
                    break
        finally:
            self._controller.clear_handle(self._job_id)
            self._controller.schedule_cleanup(
                self._job_id, delay_seconds=self._cleanup_delay_seconds
            )
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                except Exception:
                    pass


async def forward_pipeline_events(
    *,
    controller: DashboardJobController,
    job_id: str,
    event_queue: Any,
    client_queue: Queue,
    cleanup_delay_seconds: float,
) -> None:
    forwarder = PipelineEventForwarder(
        controller=controller,
        job_id=job_id,
        event_queue=event_queue,
        client_queue=client_queue,
        cleanup_delay_seconds=cleanup_delay_seconds,
    )
    await forwarder.run()
