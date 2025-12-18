from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio


pytestmark = [pytest.mark.asyncio]


@pytest_asyncio.fixture()
async def dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pipeline_dir = tmp_path / "pipeline_runs_cs"
    pipeline_dir.mkdir()
    run_dir = pipeline_dir / "run_demo"
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    (run_dir / "backtest_portfolio_csvs").mkdir(parents=True, exist_ok=True)
    (run_dir / "SUMMARY.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    (run_dir / "meta" / "ui_context.json").write_text(json.dumps({"job_id": "demo"}), encoding="utf-8")
    (run_dir / "backtest_portfolio_csvs" / "backtest_summary_top1.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (run_dir / "notes.bin").write_bytes(b"\x00\x01")

    monkeypatch.setenv("AE_PIPELINE_DIR", str(pipeline_dir))
    importlib.reload(importlib.import_module("alpha_evolve.dashboard.api.helpers"))
    importlib.reload(importlib.import_module("alpha_evolve.dashboard.api.routes.runs"))
    app_mod = importlib.reload(importlib.import_module("alpha_evolve.dashboard.api.app"))
    transport = httpx.ASGITransport(app=app_mod.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield SimpleNamespace(client=client, run_dir=run_dir, pipeline_dir=pipeline_dir)


async def test_run_assets_lists_previewable_files(dashboard):
    resp = await dashboard.client.get("/api/run-assets", params={"run_dir": "run_demo"})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert "SUMMARY.json" in items
    assert "meta/ui_context.json" in items
    assert "backtest_portfolio_csvs/backtest_summary_top1.csv" in items
    assert "notes.bin" not in items


async def test_run_assets_respects_prefix(dashboard):
    resp = await dashboard.client.get("/api/run-assets", params={"run_dir": "run_demo", "prefix": "meta"})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert items == ["meta/ui_context.json"]


async def test_run_assets_rejects_traversal_prefix(dashboard):
    resp = await dashboard.client.get("/api/run-assets", params={"run_dir": "run_demo", "prefix": "../../"})
    assert resp.status_code == 400
