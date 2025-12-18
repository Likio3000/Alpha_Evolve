from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from alpha_evolve.dashboard.api.helpers import build_pipeline_args
from alpha_evolve.dashboard.api.jobs import STATE


pytestmark = [pytest.mark.asyncio]
if os.environ.get("SKIP_MP_TESTS") == "1":
    pytestmark.append(
        pytest.mark.skip(
            "Multiprocessing primitives unavailable in this runtime sandbox."
        )
    )


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
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "top_sharpe": 1.25,
            }
        ),
        encoding="utf-8",
    )

    meta_dir = run_dir / "meta"
    meta_dir.mkdir()
    (meta_dir / "ui_context.json").write_text(
        json.dumps(
            {
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
                "pipeline_args": [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "alpha_evolve.cli.pipeline",
                    "7",
                ],
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "evolution_config.json").write_text(
        json.dumps({"generations": 7}), encoding="utf-8"
    )
    (meta_dir / "backtest_config.json").write_text(
        json.dumps({"bt_top": 12}), encoding="utf-8"
    )

    monkeypatch.setenv("AE_PIPELINE_DIR", str(pipeline_dir))

    helpers_mod = importlib.reload(
        importlib.import_module("alpha_evolve.dashboard.api.helpers")
    )
    runs_mod = importlib.reload(
        importlib.import_module("alpha_evolve.dashboard.api.routes.runs")
    )
    app_mod = importlib.reload(
        importlib.import_module("alpha_evolve.dashboard.api.app")
    )

    # httpx 0.28 automatically issues lifespan events; default construction works.
    transport = httpx.ASGITransport(app=app_mod.create_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
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


async def test_line_to_event_strips_ansi_sequences():
    from alpha_evolve.dashboard.api.routes.run_pipeline import _line_to_event

    payload = {
        "type": "gen_progress",
        "gen": 3,
        "completed": 24,
        "total": 64,
        "best": 0.42,
        "median": 0.12,
    }
    line = "\x1b[32mPROGRESS " + json.dumps(payload) + "\x1b[0m"

    event = _line_to_event(line)
    assert event["type"] == "progress"
    assert event.get("data") == payload
    assert "raw" in event
    assert "\x1b" not in event["raw"]


async def test_line_to_event_normalizes_nonfinite_values():
    from alpha_evolve.dashboard.api.routes.run_pipeline import _line_to_event

    line = (
        "\x1b[32mPROGRESS "
        '{"type": "gen_progress", "gen": 2, "completed": 3, "total": 5, '
        '"best": -Infinity, "median": NaN, "eta_sec": Infinity, "history": [NaN, 1.0]}'
        "\x1b[0m"
    )
    event = _line_to_event(line)
    data = event["data"]
    assert data["gen"] == 2
    assert data["best"] is None
    assert data["median"] is None
    assert data["eta_sec"] is None
    assert data["history"] == [None, 1.0]
    # Should serialize cleanly for SSE dispatch
    json.dumps(event)


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
    assert (
        dashboard_env.helpers.resolve_latest_run_dir()
        == dashboard_env.run_dir.resolve()
    )


async def test_resolve_latest_run_dir_handles_absolute_pointer(dashboard_env):
    """Confirm resolve_latest_run_dir handles absolute pointers written to the LATEST file."""
    latest = dashboard_env.pipeline_dir / "LATEST"
    latest.write_text(str(dashboard_env.run_dir.resolve()), encoding="utf-8")
    assert (
        dashboard_env.helpers.resolve_latest_run_dir()
        == dashboard_env.run_dir.resolve()
    )


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
    ok_resp = await dashboard_env.client.get(
        "/api/backtest-summary", params={"run_dir": "run_demo"}
    )
    assert ok_resp.status_code == 200
    rows = ok_resp.json()
    assert rows and rows[0]["AlphaID"] == "Alpha_01"

    prefixed_resp = await dashboard_env.client.get(
        "/api/backtest-summary",
        params={"run_dir": f"pipeline_runs_cs/{dashboard_env.run_dir.name}"},
    )
    assert prefixed_resp.status_code == 200
    prefixed_rows = prefixed_resp.json()
    assert prefixed_rows and prefixed_rows[0]["AlphaID"] == "Alpha_01"

    bad_resp = await dashboard_env.client.get(
        "/api/backtest-summary", params={"run_dir": "../../etc"}
    )
    assert bad_resp.status_code == 400


async def test_alpha_timeseries_endpoint(dashboard_env):
    """Exercise /api/alpha-timeseries to return aligned date/equity/ret_net vectors."""
    ok_resp = await dashboard_env.client.get(
        "/api/alpha-timeseries", params={"run_dir": "run_demo", "alpha_id": "Alpha_01"}
    )
    assert ok_resp.status_code == 200
    payload = ok_resp.json()
    assert payload["date"]
    assert len(payload["equity"]) == len(payload["date"])


async def test_alpha_timeseries_pending_when_summary_missing(
    dashboard_env, monkeypatch
):
    """Pending runs should return a 202 with empty vectors instead of surfacing errors."""
    monkeypatch.setattr(
        "alpha_evolve.dashboard.api.routes.runs._summary_csv", lambda *_a, **_k: None
    )
    resp = await dashboard_env.client.get(
        "/api/alpha-timeseries", params={"run_dir": "run_demo", "alpha_id": "Alpha_01"}
    )
    assert resp.status_code == 202
    payload = resp.json()
    assert payload["pending"] is True
    assert payload["date"] == []
    assert payload["equity"] == []
    assert payload["ret_net"] == []


async def test_run_details_endpoint_returns_metadata(dashboard_env):
    """Ensure /api/run-details exposes summary/meta payloads for the selected run."""
    resp = await dashboard_env.client.get(
        "/api/run-details", params={"run_dir": "run_demo"}
    )
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
            self._thread = threading.Thread(
                target=self._target, args=self._args, daemon=True
            )
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
        "alpha_evolve.dashboard.api.routes.run_pipeline.mp.get_context",
        lambda *_a, **_k: DummyContext(),
    )
    monkeypatch.setattr(
        "alpha_evolve.dashboard.api.routes.run_pipeline._pipeline_worker",
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


async def test_pipeline_stop_and_activity_endpoint(dashboard_env):
    """Confirm helper endpoints enforce method semantics and activity snapshots handle missing jobs."""
    stop_get = await dashboard_env.client.get("/api/pipeline/stop/unknown")
    assert stop_get.status_code == 405

    stop_post = await dashboard_env.client.post("/api/pipeline/stop/unknown")
    assert stop_post.status_code == 404

    activity_resp = await dashboard_env.client.get("/api/job-activity/unknown")
    assert activity_resp.status_code == 200
    payload = activity_resp.json()
    assert payload["exists"] is False
    assert payload["running"] is False
    assert payload["log"] == ""


async def test_pipeline_activity_snapshot_tracks_events(dashboard_env, tmp_path):
    """Ensure activity snapshots capture logs, progress, scores, and status transitions."""
    from alpha_evolve.dashboard.api.routes.run_pipeline import _forward_events

    job_id = "job-test-activity"
    client_queue = STATE.new_queue(job_id)
    event_queue: queue.Queue = queue.Queue()
    log_file = tmp_path / "dashboard" / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    STATE.init_activity(job_id, {"status": "running"})
    STATE.update_activity(job_id, log_path=str(log_file))

    async def drive_forwarder():
        await asyncio.wait_for(
            _forward_events(job_id, event_queue, client_queue), timeout=1.5
        )

    task = asyncio.create_task(drive_forwarder())
    try:
        summary_payload = {
            "type": "gen_summary",
            "generation": 1,
            "generations_total": 2,
            "pct_complete": 0.5,
            "best": {
                "fitness": 0.5,
                "fitness_static": 0.45,
                "mean_ic": 0.12,
                "ic_std": 0.03,
                "turnover": 0.1,
                "sharpe_proxy": 0.6,
                "sortino": 0.0,
                "drawdown": 0.02,
                "downside_deviation": 0.01,
                "cvar": -0.01,
                "factor_penalty": 0.0,
                "fingerprint": "fp-demo",
                "program_size": 12,
                "program": "alpha()",
                "horizon_metrics": {},
                "factor_exposures": {},
                "regime_exposures": {},
                "transaction_costs": {},
                "stress_metrics": {},
            },
            "penalties": {"parsimony": 0.01},
            "fitness_breakdown": {"parsimony": 0.01},
            "timing": {
                "generation_seconds": 1.0,
                "average_seconds": 1.0,
                "eta_seconds": 2.0,
            },
            "population": {"size": 32, "unique_fingerprints": 30},
        }

        event_queue.put({"type": "log", "raw": "Booting pipeline"})
        event_queue.put(
            {
                "type": "progress",
                "subtype": "gen_progress",
                "data": {"gen": 1, "completed": 32},
            }
        )
        event_queue.put(
            {"type": "progress", "subtype": "gen_summary", "data": summary_payload}
        )
        event_queue.put(
            {"type": "score", "sharpe_best": 1.42, "raw": "Sharpe(best)=1.42"}
        )
        event_queue.put({"type": "status", "msg": "exit", "code": 0})
        event_queue.put({"type": "__complete__"})

        await asyncio.sleep(0.05)
        activity = STATE.get_activity(job_id)
        assert activity is not None
        assert activity.get("status") == "complete"
        assert activity.get("sharpe_best") == pytest.approx(1.42)
        assert activity.get("summaries"), (
            "Expected summaries to be captured in activity snapshot"
        )

        response = await dashboard_env.client.get(f"/api/job-activity/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["exists"] is True
        assert payload["running"] is False
        assert "Booting pipeline" in payload["log"]
        assert payload["status"] == "complete"
        assert payload["sharpe_best"] == pytest.approx(1.42)
        assert payload["summaries"], (
            "Expected summaries to be returned in activity payload"
        )
        assert payload["log_path"].endswith("pipeline.log")
    finally:
        event_queue.put("__complete__")
        await asyncio.wait_for(task, timeout=1.0)
        STATE.logs.pop(job_id, None)
        STATE.queues.pop(job_id, None)
        STATE.activity.pop(job_id, None)


async def test_job_activity_reads_log_file_tail(dashboard_env, tmp_path):
    """Verify job activity falls back to the on-disk log when in-memory buffer is empty."""
    job_id = "job-log-tail"
    STATE.init_activity(job_id, {"status": "running"})
    log_file = tmp_path / "logs" / "tail.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("line-1\nline-2\n", encoding="utf-8")
    STATE.update_activity(job_id, log_path=str(log_file))

    resp = await dashboard_env.client.get(f"/api/job-activity/{job_id}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["exists"] is True
    assert payload["log"].endswith("line-2\n")

    STATE.activity.pop(job_id, None)


async def test_progress_summary_events_persist(tmp_path):
    """Ensure PROGRESS gen_summary events are broadcast and persisted to gen_summary.jsonl."""
    from alpha_evolve.dashboard.api.routes.run_pipeline import _forward_events

    job_id = "job-progress-summary"
    client_queue = STATE.new_queue(job_id)
    event_queue: queue.Queue = queue.Queue()
    STATE.set_meta(job_id, {"payload": {"generations": 3}})

    run_dir = tmp_path / "progress_run"
    run_dir.mkdir()

    summary_payload = {
        "type": "gen_summary",
        "generation": 1,
        "generations_total": 4,
        "pct_complete": 0.25,
        "best": {
            "fitness": 0.42,
            "fitness_static": 0.41,
            "mean_ic": 0.18,
            "ic_std": 0.03,
            "turnover": 0.12,
            "sharpe_proxy": 0.55,
            "sortino": 0.0,
            "drawdown": 0.01,
            "downside_deviation": 0.02,
            "cvar": -0.03,
            "factor_penalty": 0.0,
            "fingerprint": "fp-demo",
            "program_size": 18,
            "program": "alpha_prog_demo()",
            "horizon_metrics": {"1": {"mean_ic": 0.18, "ic_std": 0.03}},
            "factor_exposures": {},
            "regime_exposures": {},
            "transaction_costs": {},
            "stress_metrics": {},
        },
        "penalties": {"parsimony": 0.01},
        "fitness_breakdown": {
            "base_ic": 0.18,
            "parsimony_penalty": 0.01,
            "result": 0.17,
        },
        "timing": {
            "generation_seconds": 1.2,
            "average_seconds": 1.2,
            "eta_seconds": 6.0,
        },
        "population": {"size": 64, "unique_fingerprints": 52},
    }

    async def drive_forwarder():
        await asyncio.wait_for(
            _forward_events(job_id, event_queue, client_queue), timeout=1.0
        )

    task = asyncio.create_task(drive_forwarder())
    try:
        event_queue.put(
            {"type": "progress", "subtype": "gen_summary", "data": summary_payload}
        )
        event_queue.put({"type": "final", "run_dir": str(run_dir)})
        event_queue.put({"type": "status", "msg": "exit", "code": 0})
        event_queue.put({"type": "__complete__"})
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        STATE.logs.pop(job_id, None)
        STATE.queues.pop(job_id, None)

    streamed = []
    while True:
        try:
            streamed.append(json.loads(client_queue.get_nowait()))
        except queue.Empty:
            break
    progress_events = [item for item in streamed if item.get("type") == "progress"]
    assert progress_events, "Expected at least one progress event"
    assert progress_events[0].get("subtype") == "gen_summary"
    assert progress_events[0]["data"]["generation"] == 1

    assert job_id not in STATE.meta

    meta_dir = run_dir / "meta"
    ui_path = meta_dir / "ui_context.json"
    summary_path = meta_dir / "gen_summary.jsonl"
    assert ui_path.exists()
    assert summary_path.exists()

    contents = [
        json.loads(line)
        for line in summary_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert contents and contents[0]["generation"] == 1
