from __future__ import annotations

import importlib
from types import SimpleNamespace
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


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


@pytest.fixture()
def dashboard_env(tmp_path, monkeypatch):
    pipeline_dir = tmp_path / "pipeline_runs_cs"
    pipeline_dir.mkdir()
    run_dir = pipeline_dir / "run_demo"
    run_dir.mkdir()
    _write_backtest_files(run_dir)
    (pipeline_dir / "LATEST").write_text("run_demo", encoding="utf-8")

    monkeypatch.setenv("AE_PIPELINE_DIR", str(pipeline_dir))

    helpers_mod = importlib.reload(importlib.import_module("scripts.dashboard_server.helpers"))
    runs_mod = importlib.reload(importlib.import_module("scripts.dashboard_server.routes.runs"))
    app_mod = importlib.reload(importlib.import_module("scripts.dashboard_server.app"))

    client = TestClient(app_mod.create_app())

    try:
        yield SimpleNamespace(
            client=client,
            helpers=helpers_mod,
            pipeline_dir=pipeline_dir,
            run_dir=run_dir,
        )
    finally:
        client.close()
        # Reload modules so subsequent tests see default environment-derived paths
        importlib.reload(helpers_mod)
        importlib.reload(runs_mod)
        importlib.reload(app_mod)


def test_resolve_latest_run_dir_reads_relative_pointer(dashboard_env):
    assert dashboard_env.helpers.resolve_latest_run_dir() == dashboard_env.run_dir.resolve()


def test_resolve_latest_run_dir_handles_absolute_pointer(dashboard_env):
    latest = dashboard_env.pipeline_dir / "LATEST"
    latest.write_text(str(dashboard_env.run_dir.resolve()), encoding="utf-8")
    assert dashboard_env.helpers.resolve_latest_run_dir() == dashboard_env.run_dir.resolve()


def test_last_run_endpoint_returns_expected_payload(dashboard_env):
    resp = dashboard_env.client.get("/api/last-run")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_dir"] == "run_demo"
    assert payload["sharpe_best"] == pytest.approx(1.25)


def test_runs_endpoint_lists_runs(dashboard_env):
    resp = dashboard_env.client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["path"] == "run_demo"


def test_backtest_summary_requires_valid_run_dir(dashboard_env):
    ok_resp = dashboard_env.client.get("/api/backtest-summary", params={"run_dir": "run_demo"})
    assert ok_resp.status_code == 200
    rows = ok_resp.json()
    assert rows and rows[0]["AlphaID"] == "Alpha_01"

    bad_resp = dashboard_env.client.get("/api/backtest-summary", params={"run_dir": "../../etc"})
    assert bad_resp.status_code == 400


def test_alpha_timeseries_endpoint(dashboard_env):
    ok_resp = dashboard_env.client.get("/api/alpha-timeseries", params={"run_dir": "run_demo", "alpha_id": "Alpha_01"})
    assert ok_resp.status_code == 200
    payload = ok_resp.json()
    assert payload["date"]
    assert len(payload["equity"]) == len(payload["date"])


def test_pipeline_run_does_not_block_healthcheck(dashboard_env, monkeypatch):
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

    resp = dashboard_env.client.post("/api/pipeline/run", json={"generations": 1})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert iter_started.wait(timeout=1.0)

    try:
        health_resp = dashboard_env.client.get("/health", timeout=1.0)
    finally:
        release.set()

    assert health_resp.status_code == 200
    assert health_resp.json()["ok"] is True

    # Ensure the background job plumbing settles before the fixture teardown
    for _ in range(20):
        status_resp = dashboard_env.client.get(f"/api/job-status/{job_id}")
        payload = status_resp.json()
        if not payload.get("running"):
            break
        time.sleep(0.05)
