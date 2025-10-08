from __future__ import annotations

import asyncio
import importlib
import json
import queue
from types import SimpleNamespace
import threading
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from scripts.dashboard_server.helpers import build_pipeline_args, ROOT
from scripts.dashboard_server.jobs import STATE


pytestmark = pytest.mark.asyncio

def _write_backtest_files(run_dir: Path) -> None:
    bt_dir = run_dir / "backtest_portfolio_csvs"
    bt_dir.mkdir()
    summary = bt_dir / "backtest_summary_top1.csv"
    summary.write_text(
        "AlphaID,TS,TimeseriesFile,Sharpe,AnnReturn,AnnVol,MaxDD,Turnover,Ops,Program\n"
        "Alpha_01,alpha_01.csv,alpha_01.csv,1.25,0.4,0.2,-0.1,0.5,12,prog\n",
        encoding="utf-8",
    )
    ts_file = bt_dir / "alpha_01.csv"
    ts_file.write_text(
        "date,equity,ret_net\n2020-01-01,100,0.01\n2020-01-02,101,0.00\n",
        encoding="utf-8",
    )


@pytest_asyncio.fixture()
async def dashboard_env(tmp_path, monkeypatch):
    pipeline_dir = tmp_path / "pipeline_runs_cs"
    pipeline_dir.mkdir()
    run_dir = pipeline_dir / "run_demo"
    run_dir.mkdir()
    _write_backtest_files(run_dir)
    (pipeline_dir / "LATEST").write_text("run_demo", encoding="utf-8")

    summary_path = run_dir / "SUMMARY.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1,
        "top_sharpe": 1.25,
    }), encoding="utf-8")

    meta_dir = run_dir / "meta"
    meta_dir.mkdir()
    (meta_dir / "ui_context.json").write_text(json.dumps({
        "job_id": "abc123",
        "submitted_at": "2024-01-01T00:00:00Z",
        "payload": {
            "generations": 7,
            "dataset": "sp500",
            "overrides": {
                "bt_top": 12,
                "dry_run": False,
            },
        },
        "pipeline_args": ["uv", "run", "run_pipeline.py", "7"],
    }), encoding="utf-8")
    (meta_dir / "evolution_config.json").write_text(json.dumps({"generations": 7}), encoding="utf-8")
    (meta_dir / "backtest_config.json").write_text(json.dumps({"bt_top": 12}), encoding="utf-8")

    monkeypatch.setenv("AE_PIPELINE_DIR", str(pipeline_dir))

    helpers_mod = importlib.reload(importlib.import_module("scripts.dashboard_server.helpers"))
    runs_mod = importlib.reload(importlib.import_module("scripts.dashboard_server.routes.runs"))
    app_mod = importlib.reload(importlib.import_module("scripts.dashboard_server.app"))


    # httpx 0.28 automatically issues lifespan events; default construction works.
    transport = httpx.ASGITransport(app=app_mod.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        try:
            yield SimpleNamespace(
                client=client,
                helpers=helpers_mod,
                pipeline_dir=pipeline_dir,
                run_dir=run_dir,
            )
        finally:
            # Reload modules so subsequent tests see default environment-derived paths
            importlib.reload(helpers_mod)
            importlib.reload(runs_mod)
            importlib.reload(app_mod)


async def test_build_pipeline_args_defaults_to_sp500():
    payload = {"generations": 3, "dataset": "sp500"}
    args = build_pipeline_args(payload, include_runner=False)
    assert args[0] == "3"
    assert args[1] == "--config"
    assert args[2].endswith("configs/sp500.toml")

    args_no_dataset = build_pipeline_args({"generations": 2}, include_runner=False)
    assert "--config" not in args_no_dataset


async def test_resolve_latest_run_dir_reads_relative_pointer(dashboard_env):
    """Ensure resolve_latest_run_dir reads the relative pointer from LATEST and returns the run path."""
    assert dashboard_env.helpers.resolve_latest_run_dir() == dashboard_env.run_dir.resolve()


async def test_resolve_latest_run_dir_handles_absolute_pointer(dashboard_env):
    """Confirm resolve_latest_run_dir handles absolute pointers written to the LATEST file."""
    latest = dashboard_env.pipeline_dir / "LATEST"
    latest.write_text(str(dashboard_env.run_dir.resolve()), encoding="utf-8")
    assert dashboard_env.helpers.resolve_latest_run_dir() == dashboard_env.run_dir.resolve()


async def test_last_run_endpoint_returns_expected_payload(dashboard_env):
    """Verify /api/last-run reports the latest pipeline directory and Sharpe summary."""
    resp = await dashboard_env.client.get("/api/last-run")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_dir"] == "run_demo"
    assert payload["sharpe_best"] == pytest.approx(1.25)


async def test_runs_endpoint_lists_runs(dashboard_env):
    """Check /api/runs includes the seeded run with its relative path metadata."""
    resp = await dashboard_env.client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["path"] == "run_demo"


async def test_backtest_summary_requires_valid_run_dir(dashboard_env):
    """Confirm /api/backtest-summary returns rows for a valid run and rejects traversal attempts."""
    ok_resp = await dashboard_env.client.get("/api/backtest-summary", params={"run_dir": "run_demo"})
    assert ok_resp.status_code == 200
    rows = ok_resp.json()
    assert rows and rows[0]["AlphaID"] == "Alpha_01"

    bad_resp = await dashboard_env.client.get("/api/backtest-summary", params={"run_dir": "../../etc"})
    assert bad_resp.status_code == 400


async def test_alpha_timeseries_endpoint(dashboard_env):
    """Exercise /api/alpha-timeseries to return aligned date/equity/ret_net vectors."""
    ok_resp = await dashboard_env.client.get("/api/alpha-timeseries", params={"run_dir": "run_demo", "alpha_id": "Alpha_01"})
    assert ok_resp.status_code == 200
    payload = ok_resp.json()
    assert payload["date"]
    assert len(payload["equity"]) == len(payload["date"])


async def test_run_details_endpoint_returns_metadata(dashboard_env):
    """Ensure /api/run-details exposes summary/meta payloads for the selected run."""
    resp = await dashboard_env.client.get("/api/run-details", params={"run_dir": "run_demo"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["name"] == "run_demo"
    assert payload["sharpe_best"] == pytest.approx(1.25)
    assert payload["summary"]["schema_version"] == 1
    assert payload["ui_context"]["payload"]["generations"] == 7
    assert payload["meta"]["evolution_config"]["generations"] == 7


async def test_pipeline_run_does_not_block_healthcheck(dashboard_env, monkeypatch):
    """Simulate a long pipeline run and ensure the health endpoint remains responsive."""
    release = threading.Event()
    iter_started = threading.Event()

    def fake_worker(cli_args, root_dir, event_queue):
        iter_started.set()
        try:
            if not release.wait(timeout=3.0):
                event_queue.put({"type": "error", "code": 1, "detail": "timeout"})
                event_queue.put({"type": "status", "msg": "exit", "code": 1})
                return
        finally:
            release.set()
        event_queue.put({"type": "status", "msg": "exit", "code": 0})
        event_queue.put({"type": "__complete__"})

    class DummyProcess:
        def __init__(self, target, args, daemon=True):
            self._thread: threading.Thread | None = None
            self._target = target
            self._args = args

        def start(self):
            self._thread = threading.Thread(target=self._target, args=self._args, daemon=True)
            self._thread.start()

        def is_alive(self):
            return self._thread is not None and self._thread.is_alive()

        def terminate(self):
            release.set()

    class DummyContext:
        def Queue(self):
            return queue.Queue()

        def Process(self, target, args, daemon=True):
            return DummyProcess(lambda: fake_worker(*args), (), daemon=daemon)

    monkeypatch.setattr(
        "scripts.dashboard_server.routes.run_pipeline.mp.get_context",
        lambda *_a, **_k: DummyContext(),
    )
    monkeypatch.setattr(
        "scripts.dashboard_server.routes.run_pipeline._pipeline_worker",
        fake_worker,
    )

    resp = await dashboard_env.client.post("/api/pipeline/run", json={"generations": 1})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert iter_started.wait(timeout=1.0)

    try:
        health_resp = await dashboard_env.client.get("/health", timeout=1.0)
    finally:
        release.set()

    assert health_resp.status_code == 200
    assert health_resp.json()["ok"] is True

    # Ensure the background job plumbing settles before the fixture teardown
    for _ in range(20):
        status_resp = await dashboard_env.client.get(f"/api/job-status/{job_id}")
        payload = status_resp.json()
        if not payload.get("running"):
            break
        await asyncio.sleep(0.05)


async def test_pipeline_stop_and_events_endpoints(dashboard_env):
    """Confirm new pipeline helper endpoints enforce method semantics and report missing jobs."""
    stop_get = await dashboard_env.client.get("/api/pipeline/stop/unknown")
    assert stop_get.status_code == 405

    stop_post = await dashboard_env.client.post("/api/pipeline/stop/unknown")
    assert stop_post.status_code == 404

    events_resp = await dashboard_env.client.get("/api/pipeline/events/unknown")
    assert events_resp.status_code == 404


async def test_pipeline_events_streams_incremental_logs(dashboard_env, monkeypatch):
    """Ensure SSE scaffolding streams log and score payloads before exit."""
    from scripts.dashboard_server.routes.run_pipeline import _forward_events
    from scripts.dashboard_server.helpers import _SSEStream

    job_id = "job-test-stream"
    client_queue = STATE.new_queue(job_id)
    event_queue: queue.Queue = queue.Queue()

    async def drive_forwarder():
        await asyncio.wait_for(_forward_events(job_id, event_queue, client_queue), timeout=1.5)

    task = asyncio.create_task(drive_forwarder())
    try:
        event_queue.put({"type": "log", "raw": "Booting pipeline"})
        event_queue.put({"type": "score", "sharpe_best": 1.42, "raw": "Sharpe(best)=1.42"})
        event_queue.put({"type": "status", "msg": "exit", "code": 0})
        event_queue.put({"type": "__complete__"})

        await asyncio.sleep(0.05)
        log_text = STATE.get_log_text(job_id)
        assert "Booting pipeline" in log_text

        streamed = []
        while True:
            try:
                streamed.append(json.loads(client_queue.get_nowait()))
            except queue.Empty:
                break
        assert any(item["type"] == "log" for item in streamed)
        assert any(item["type"] == "score" and pytest.approx(1.42) == item["sharpe_best"] for item in streamed)
        assert any(item["type"] == "status" and item.get("msg") == "exit" for item in streamed)

        q = queue.Queue()
        sse = _SSEStream(q, keepalive=0.05, prefer_async=False)
        q.put(json.dumps({"type": "test", "payload": 1}))
        first = next(iter(sse))
        assert first.startswith("data: {") and '"type": "test"' in first
    finally:
        event_queue.put("__complete__")
        await asyncio.wait_for(task, timeout=1.0)
        STATE.logs.pop(job_id, None)
        STATE.queues.pop(job_id, None)
