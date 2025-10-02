from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace
import threading
from pathlib import Path

import httpx
import pytest
import pytest_asyncio


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

    monkeypatch.setenv("AE_PIPELINE_DIR", str(pipeline_dir))

    helpers_mod = importlib.reload(importlib.import_module("scripts.dashboard_server.helpers"))
    runs_mod = importlib.reload(importlib.import_module("scripts.dashboard_server.routes.runs"))
    selfplay_mod = importlib.reload(importlib.import_module("scripts.dashboard_server.routes.selfplay"))
    app_mod = importlib.reload(importlib.import_module("scripts.dashboard_server.app"))

    transport = httpx.ASGITransport(app=app_mod.create_app(), lifespan="auto")
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
            importlib.reload(selfplay_mod)
            importlib.reload(app_mod)


async def test_resolve_latest_run_dir_reads_relative_pointer(dashboard_env):
    assert dashboard_env.helpers.resolve_latest_run_dir() == dashboard_env.run_dir.resolve()


async def test_resolve_latest_run_dir_handles_absolute_pointer(dashboard_env):
    latest = dashboard_env.pipeline_dir / "LATEST"
    latest.write_text(str(dashboard_env.run_dir.resolve()), encoding="utf-8")
    assert dashboard_env.helpers.resolve_latest_run_dir() == dashboard_env.run_dir.resolve()


async def test_last_run_endpoint_returns_expected_payload(dashboard_env):
    resp = await dashboard_env.client.get("/api/last-run")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_dir"] == "run_demo"
    assert payload["sharpe_best"] == pytest.approx(1.25)


async def test_runs_endpoint_lists_runs(dashboard_env):
    resp = await dashboard_env.client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["path"] == "run_demo"


async def test_backtest_summary_requires_valid_run_dir(dashboard_env):
    ok_resp = await dashboard_env.client.get("/api/backtest-summary", params={"run_dir": "run_demo"})
    assert ok_resp.status_code == 200
    rows = ok_resp.json()
    assert rows and rows[0]["AlphaID"] == "Alpha_01"

    bad_resp = await dashboard_env.client.get("/api/backtest-summary", params={"run_dir": "../../etc"})
    assert bad_resp.status_code == 400


async def test_alpha_timeseries_endpoint(dashboard_env):
    ok_resp = await dashboard_env.client.get("/api/alpha-timeseries", params={"run_dir": "run_demo", "alpha_id": "Alpha_01"})
    assert ok_resp.status_code == 200
    payload = ok_resp.json()
    assert payload["date"]
    assert len(payload["equity"]) == len(payload["date"])


async def test_pipeline_run_does_not_block_healthcheck(dashboard_env, monkeypatch):
    release = threading.Event()
    iter_started = threading.Event()
    lines = ["DIAG {\"foo\": 1}\n"]

    class BlockingStdout:
        def __iter__(self):
            return self

        def __next__(self):
            iter_started.set()
            if not release.wait(timeout=3.0):
                raise RuntimeError("test release event was not set")
            if lines:
                return lines.pop(0)
            raise StopIteration

    class DummyPopen:
        def __init__(self):
            self.stdout = BlockingStdout()
            self._returncode = None

        def wait(self):
            if self._returncode is None:
                self._returncode = 0
            return self._returncode

        def poll(self):
            return self._returncode

        def terminate(self):
            release.set()
            self._returncode = -15

        def kill(self):
            release.set()
            self._returncode = -9

    def fake_popen(*args, **kwargs):
        return DummyPopen()

    monkeypatch.setattr(
        "scripts.dashboard_server.routes.run_pipeline.subprocess.Popen",
        fake_popen,
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


def _write_selfplay_session(root: Path) -> Path:
    session_root = root / "self_evolution"
    session_root.mkdir(exist_ok=True)
    session_dir = session_root / "self_evo_session_20250101_000001"
    session_dir.mkdir(parents=True, exist_ok=True)

    (session_dir / "history.jsonl").write_text("", encoding="utf-8")
    briefings = [
        {
            "timestamp": "2025-01-01T00:00:05Z",
            "iteration": 0,
            "objective_metric": "Sharpe",
            "analysis": {"trend": "improving", "objective": 1.2, "alpha_count": 3},
        },
        {
            "timestamp": "2025-01-01T00:10:05Z",
            "iteration": 1,
            "objective_metric": "Sharpe",
            "analysis": {"trend": "flat", "objective": 1.18, "alpha_count": 2},
        },
    ]
    with (session_dir / "agent_briefings.jsonl").open("w", encoding="utf-8") as fh:
        for entry in briefings:
            fh.write(json.dumps(entry) + "\n")

    pending = {
        "timestamp": "2025-01-01T00:10:10Z",
        "iteration_completed": 1,
        "status": "awaiting_approval",
        "summary": "Trend: flat | Objective: 1.1800 | Alphas: 2",
    }
    (session_dir / "pending_action.json").write_text(json.dumps(pending, indent=2), encoding="utf-8")

    summary = {
        "objective_metric": "Sharpe",
        "maximize": True,
        "iterations": 2,
        "history_file": str(session_dir / "history.jsonl"),
    }
    (session_dir / "session_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return session_dir


async def test_selfplay_status_endpoint_returns_latest_session(dashboard_env):
    session_dir = _write_selfplay_session(dashboard_env.pipeline_dir)

    resp = await dashboard_env.client.get("/api/selfplay/status")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["session_name"] == session_dir.name
    assert payload["pending_action"]["status"] == "awaiting_approval"
    assert len(payload["briefings"]) == 2


async def test_selfplay_approval_endpoint_updates_file(dashboard_env):
    session_dir = _write_selfplay_session(dashboard_env.pipeline_dir)

    resp = await dashboard_env.client.post(
        "/api/selfplay/approval",
        json={"status": "approved"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["pending_action"]["status"] == "approved"

    updated = json.loads((session_dir / "pending_action.json").read_text(encoding="utf-8"))
    assert updated["status"] == "approved"


async def test_selfplay_run_endpoint_starts_job(dashboard_env, monkeypatch):
    search_space = dashboard_env.helpers.ROOT / "configs" / "self_evolution" / "sample_crypto_space.json"

    class DummyPopen:
        def __init__(self, *args, **kwargs):
            self._lines = ["Iteration 0 objective=0.10\n", "done\n"]
            self.stdout = iter(self._lines)
            self._returncode = 0

        def wait(self):
            return self._returncode

        def poll(self):
            return None

        def terminate(self):
            self._returncode = -15

        def kill(self):
            self._returncode = -9

    monkeypatch.setattr(
        "scripts.dashboard_server.routes.selfplay.subprocess.Popen",
        lambda *a, **kw: DummyPopen(),
    )

    resp = await dashboard_env.client.post(
        "/api/selfplay/run",
        json={
            "search_space": str(search_space),
            "iterations": 1,
            "auto_approve": True,
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id

    for _ in range(20):
        status_resp = await dashboard_env.client.get(f"/api/job-status/{job_id}")
        status = status_resp.json()
        if not status.get("running"):
            break
        await asyncio.sleep(0.01)

    log_resp = await dashboard_env.client.get(f"/api/job-log/{job_id}")
    log_payload = log_resp.json()
    assert "log" in log_payload
