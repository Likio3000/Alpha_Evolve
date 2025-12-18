from __future__ import annotations

import asyncio
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
import pytest_asyncio

from alpha_evolve.dashboard.api.jobs import STATE


pytestmark = [pytest.mark.asyncio]


def _reset_state() -> None:
    STATE.queues.clear()
    STATE.handles.clear()
    STATE.logs.clear()
    STATE.meta.clear()
    STATE.activity.clear()


@pytest_asyncio.fixture()
async def dashboard_exp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pipeline_dir = tmp_path / "pipeline_runs_cs"
    pipeline_dir.mkdir()
    monkeypatch.setenv("AE_PIPELINE_DIR", str(pipeline_dir))
    monkeypatch.setenv("AE_EXPERIMENTS_DB", str(tmp_path / "experiments.sqlite"))

    helpers_mod = importlib.reload(importlib.import_module("alpha_evolve.dashboard.api.helpers"))
    exp_routes_mod = importlib.reload(importlib.import_module("alpha_evolve.dashboard.api.routes.experiments"))
    app_mod = importlib.reload(importlib.import_module("alpha_evolve.dashboard.api.app"))

    transport = httpx.ASGITransport(app=app_mod.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        try:
            yield SimpleNamespace(
                client=client,
                pipeline_dir=pipeline_dir,
                helpers=helpers_mod,
                routes=exp_routes_mod,
            )
        finally:
            _reset_state()


async def test_dashboard_experiment_session_approval_and_export(
    dashboard_exp_env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # Stub out the inner pipeline runner so we don't run the full evolution loop.
    def stub_pipeline_runner(evo_cfg, bt_cfg, options):  # noqa: ANN001
        idx = stub_pipeline_runner.counter
        run_dir = Path(options.output_dir) / f"run_exp_{idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        bt_dir = run_dir / "backtest_portfolio_csvs"
        bt_dir.mkdir(parents=True, exist_ok=True)
        members = ["Alpha_01", "Alpha_02", "Alpha_03"]
        pd.DataFrame(
            [[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]],
            index=members,
            columns=members,
        ).to_csv(bt_dir / "return_corr_matrix.csv")
        (bt_dir / "ensemble_selection.json").write_text(json.dumps({"members": members}), encoding="utf-8")

        sharpe = 1.0 + 0.1 * idx
        (run_dir / "SUMMARY.json").write_text(
            json.dumps({"best_metrics": {"Sharpe": sharpe}, "backtested_alphas": 3}),
            encoding="utf-8",
        )
        stub_pipeline_runner.counter += 1
        return run_dir

    stub_pipeline_runner.counter = 0

    from alpha_evolve.experiments.runner import ExperimentSessionRunner as RealRunner

    class PatchedRunner(RealRunner):
        def __init__(self, *args, **kwargs):  # noqa: ANN001
            kwargs["pipeline_runner"] = stub_pipeline_runner
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(dashboard_exp_env.routes, "ExperimentSessionRunner", PatchedRunner)

    space_path = tmp_path / "space.json"
    space_path.write_text(
        json.dumps(
            {
                "parameters": [
                    {"key": "evolution.pop_size", "type": "choice", "values": [120], "mutate_probability": 1.0}
                ]
            }
        ),
        encoding="utf-8",
    )

    resp = await dashboard_exp_env.client.post(
        "/api/experiments/start",
        json={
            "search_space": str(space_path),
            "config": None,
            "iterations": 2,
            "seed": 0,
            "auto_approve": False,
            "approval_poll_interval": 0.01,
        },
    )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    # Wait for the proposal to appear (runner blocks on approval after iteration 0).
    async def _pending_proposal_id() -> int | None:
        res = await dashboard_exp_env.client.get(f"/api/experiments/sessions/{session_id}/proposals")
        if res.status_code != 200:
            return None
        items = res.json().get("items") or []
        for item in items:
            if item.get("status") == "pending":
                return int(item["id"])
        return None

    proposal_id = None
    start = time.time()
    while proposal_id is None and time.time() - start < 5.0:
        proposal_id = await _pending_proposal_id()
        if proposal_id is None:
            await asyncio.sleep(0.05)
    assert proposal_id is not None

    decision_resp = await dashboard_exp_env.client.post(
        f"/api/experiments/sessions/{session_id}/proposals/{proposal_id}/decision",
        json={"decision": "approved", "decided_by": "test"},
    )
    assert decision_resp.status_code == 200
    assert decision_resp.json()["ok"] is True

    # Runner should now complete iteration 1.
    async def _is_completed() -> bool:
        s = await dashboard_exp_env.client.get(f"/api/experiments/sessions/{session_id}")
        if s.status_code != 200:
            return False
        return s.json().get("status") == "completed"

    start = time.time()
    while time.time() - start < 5.0:
        if await _is_completed():
            break
        await asyncio.sleep(0.05)
    assert await _is_completed()

    export_resp = await dashboard_exp_env.client.post(
        f"/api/experiments/sessions/{session_id}/export-best-config",
    )
    assert export_resp.status_code == 200
    payload = export_resp.json()
    assert "config_path" in payload
    cfg_path = Path(payload["config_path"])
    if not cfg_path.is_absolute():
        cfg_path = (dashboard_exp_env.helpers.ROOT / cfg_path).resolve()
    assert cfg_path.exists()
    import tomllib

    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert "evolution" in data
    assert "backtest" in data
