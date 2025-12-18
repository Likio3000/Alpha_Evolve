from __future__ import annotations

import asyncio
import importlib
import json
import sys
import textwrap
import threading
import queue as queue_mod
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from alpha_evolve.dashboard.api.jobs import STATE
from alpha_evolve.dashboard.api.helpers import make_sse_response


pytestmark = [pytest.mark.asyncio]


def _reset_state() -> None:
    STATE.queues.clear()
    STATE.handles.clear()
    STATE.logs.clear()
    STATE.meta.clear()
    STATE.activity.clear()


@pytest_asyncio.fixture()
async def dashboard_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pipeline_dir = tmp_path / "pipeline_runs_cs"
    pipeline_dir.mkdir()
    monkeypatch.setenv("AE_PIPELINE_DIR", str(pipeline_dir))

    helpers_mod = importlib.reload(importlib.import_module("alpha_evolve.dashboard.api.helpers"))
    run_pipeline_mod = importlib.reload(importlib.import_module("alpha_evolve.dashboard.api.routes.run_pipeline"))
    app_mod = importlib.reload(importlib.import_module("alpha_evolve.dashboard.api.app"))

    transport = httpx.ASGITransport(app=app_mod.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        try:
            yield SimpleNamespace(
                client=client,
                pipeline_dir=pipeline_dir,
                helpers=helpers_mod,
                run_pipeline=run_pipeline_mod,
            )
        finally:
            _reset_state()


async def _wait_for(predicate, *, timeout: float = 2.0, interval: float = 0.05):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(interval)
    return None


async def test_subprocess_runner_streams_sse_and_persists_meta(dashboard_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dummy_script = tmp_path / "dummy_pipeline.py"
    dummy_script.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import time
            from pathlib import Path

            pipeline_dir = Path(os.environ["AE_PIPELINE_DIR"])
            job_id = os.environ.get("PIPELINE_JOB_ID", "job")
            run_dir = pipeline_dir / f"run_{job_id.replace('-', '')[:8]}"
            (run_dir / "meta").mkdir(parents=True, exist_ok=True)
            pipeline_dir.joinpath("LATEST").write_text(run_dir.name, encoding="utf-8")

            print("Booting pipeline", flush=True)
            print("PROGRESS " + json.dumps({"type": "gen_progress", "gen": 1, "completed": 1, "total": 2}), flush=True)
            print(
                "PROGRESS "
                + json.dumps({"type": "gen_summary", "generation": 1, "generations_total": 2, "pct_complete": 0.5}),
                flush=True,
            )
            print("Sharpe(best) = 1.23", flush=True)
            time.sleep(0.1)
            """
        ).lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        dashboard_env.run_pipeline,
        "_build_subprocess_command",
        lambda _cli_args: [sys.executable, "-u", str(dummy_script)],
    )

    resp = await dashboard_env.client.post(
        "/api/pipeline/run",
        json={"generations": 1, "runner_mode": "subprocess"},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    health = await dashboard_env.client.get("/health", timeout=1.0)
    assert health.status_code == 200
    assert health.json()["ok"] is True

    latest_path = dashboard_env.pipeline_dir / "LATEST"
    run_dir = await _wait_for(lambda: latest_path.exists() and latest_path.read_text(encoding="utf-8").strip(), timeout=2.0)
    assert run_dir
    resolved_run_dir = (dashboard_env.pipeline_dir / str(run_dir)).resolve()

    ui_context_path = resolved_run_dir / "meta" / "ui_context.json"
    summary_path = resolved_run_dir / "meta" / "gen_summary.jsonl"

    await _wait_for(lambda: ui_context_path.exists() and summary_path.exists(), timeout=2.0)
    assert ui_context_path.exists()
    assert summary_path.exists()

    activity = await dashboard_env.client.get(f"/api/job-activity/{job_id}")
    assert activity.status_code == 200
    payload = activity.json()
    assert payload["exists"] is True
    assert payload["sharpe_best"] == pytest.approx(1.23)
    assert payload.get("summaries"), "Expected gen_summary to appear in job activity snapshots."

    # Validate the job queue contains JSON events suitable for SSE dispatch.
    q = STATE.get_queue(job_id)
    assert q is not None
    drained: list[dict] = []
    while True:
        try:
            drained.append(json.loads(q.get_nowait()))
        except queue_mod.Empty:
            break
    types = {item.get("type") for item in drained}
    assert "status" in types
    assert "score" in types
    assert "gen_summary" in types


async def test_multiprocessing_runner_can_stop_without_blocking(dashboard_env, monkeypatch: pytest.MonkeyPatch):
    release = threading.Event()
    started = threading.Event()

    def fake_worker(cli_args, root_dir, event_queue, *extra):  # noqa: ANN001
        started.set()
        release.wait(timeout=5.0)

    class DummyProcess:
        def __init__(self, target, args, daemon=True):  # noqa: ANN001
            self._thread = threading.Thread(target=target, args=args, daemon=daemon)

        def start(self):  # noqa: D401
            self._thread.start()

        def is_alive(self):  # noqa: D401
            return self._thread.is_alive()

        def terminate(self):  # noqa: D401
            release.set()

    class DummyContext:
        def Queue(self):  # noqa: D401
            import queue

            return queue.Queue()

        def Process(self, target, args, daemon=True):  # noqa: ANN001
            return DummyProcess(target, args, daemon=daemon)

    monkeypatch.setattr(
        "alpha_evolve.dashboard.api.routes.run_pipeline.mp.get_context",
        lambda *_a, **_k: DummyContext(),
    )
    monkeypatch.setattr(
        "alpha_evolve.dashboard.api.routes.run_pipeline._pipeline_worker",
        fake_worker,
    )

    resp = await dashboard_env.client.post(
        "/api/pipeline/run",
        json={"generations": 1, "runner_mode": "multiprocessing"},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert started.wait(timeout=1.0)

    health = await dashboard_env.client.get("/health", timeout=1.0)
    assert health.status_code == 200
    assert health.json()["ok"] is True

    stop_resp = await dashboard_env.client.post(f"/api/pipeline/stop/{job_id}")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["stopped"] is True

    async def _poll_stopped() -> bool:
        snapshot = await dashboard_env.client.get(f"/api/job-activity/{job_id}")
        data = snapshot.json()
        return bool(data.get("status") == "error" and not data.get("running", True))

    async def _wait_for_stop() -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 3.0
        while loop.time() < deadline:
            if await _poll_stopped():
                return
            await asyncio.sleep(0.05)
        raise AssertionError("Expected job activity to report stopped/error status.")

    await asyncio.wait_for(_wait_for_stop(), timeout=4.0)


async def test_make_sse_response_formats_frames():
    q: queue_mod.Queue[str] = queue_mod.Queue()
    q.put_nowait(json.dumps({"type": "status", "msg": "started"}))
    response = make_sse_response(q, keepalive_seconds=0.01)

    stream = response.streaming_content
    first = await asyncio.wait_for(anext(stream), timeout=1.0)
    assert isinstance(first, (bytes, bytearray))
    assert first.startswith(b"data: ")
    assert first.endswith(b"\n\n")

    # Empty queue -> ping frame.
    ping = await asyncio.wait_for(anext(stream), timeout=1.0)
    assert isinstance(ping, (bytes, bytearray))
    assert b"event: ping" in ping
    await stream.aclose()
