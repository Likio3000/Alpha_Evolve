from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_evolve.dashboard.api.pipeline_runs import (
    build_job_status_payload,
    build_pending_pipeline_timeseries_payload,
    build_pipeline_backtest_summary_rows,
    build_pipeline_run_details,
    build_pipeline_runs_listing,
    build_pipeline_timeseries_payload,
    list_pipeline_run_assets,
    resolve_pipeline_run_asset_path,
    resolve_pipeline_run_dir,
    resolve_pipeline_timeseries_file,
    update_pipeline_run_label,
)


def test_build_job_status_payload_tracks_missing_and_running_handles() -> None:
    class RunningHandle:
        def is_running(self) -> bool:
            return True

    assert build_job_status_payload(handle=None) == {"exists": False, "running": False}
    assert build_job_status_payload(handle=RunningHandle()) == {"exists": True, "running": True}


def test_build_pipeline_backtest_summary_rows_normalizes_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text(
        "AlphaID,TimeseriesFile,Sharpe,AnnReturn,AnnVol,MaxDD,Turnover,Ops,IC,PROGRAM\n"
        "Alpha_01,folder/demo.csv,1.5,0.2,0.1,-0.05,0.3,5,0.11,alpha()\n",
        encoding="utf-8",
    )

    rows = build_pipeline_backtest_summary_rows(csv_path)

    assert rows == [
        {
            "AlphaID": "Alpha_01",
            "TS": "demo.csv",
            "TimeseriesFile": "folder/demo.csv",
            "Sharpe": 1.5,
            "AnnReturn": 0.2,
            "AnnVol": 0.1,
            "MaxDD": -0.05,
            "Turnover": 0.3,
            "Ops": "5",
            "OriginalMetric": 0.11,
            "Program": "alpha()",
        }
    ]


def test_list_pipeline_run_assets_sorts_and_filters_previewable_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_demo"
    (run_dir / "meta").mkdir(parents=True)
    (run_dir / "meta" / "z.json").write_text("{}", encoding="utf-8")
    (run_dir / "aaa.log").write_text("ok\n", encoding="utf-8")
    (run_dir / "skip.bin").write_bytes(b"\x00")

    payload = list_pipeline_run_assets(resolved_run_dir=run_dir, prefix="", limit=10)

    assert payload == {"items": ["aaa.log", "meta/z.json"]}


def test_resolve_pipeline_timeseries_file_uses_summary_alpha_lookup(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_demo"
    backtest_dir = run_dir / "backtest_portfolio_csvs"
    backtest_dir.mkdir(parents=True)
    (backtest_dir / "backtest_summary_top1.csv").write_text(
        "AlphaID,TS\nAlpha_01,alpha_01.csv\n",
        encoding="utf-8",
    )

    resolved = resolve_pipeline_timeseries_file(run_dir=run_dir, file=None, alpha_id="Alpha_01")

    assert resolved == backtest_dir / "alpha_01.csv"


def test_build_pipeline_timeseries_payload_and_pending_shape(tmp_path: Path) -> None:
    csv_path = tmp_path / "alpha.csv"
    csv_path.write_text(
        "date,equity,ret_net\n2024-01-01,1.0,0.1\n2024-01-02,bad,0.2\n",
        encoding="utf-8",
    )

    payload = build_pipeline_timeseries_payload(csv_path)
    assert payload["date"] == ["2024-01-01", "2024-01-02"]
    assert payload["ret_net"] == [0.1, 0.2]
    assert payload["equity"][0] == 1.0
    assert payload["equity"][1] is None
    assert build_pending_pipeline_timeseries_payload() == {
        "date": [],
        "equity": [],
        "ret_net": [],
        "pending": True,
    }


def test_list_pipeline_run_assets_rejects_prefix_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_demo"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="prefix must stay within run_dir"):
        list_pipeline_run_assets(resolved_run_dir=run_dir, prefix="../..", limit=10)


def test_resolve_pipeline_run_dir_rejects_pipeline_root(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline_runs_cs"
    pipeline_dir.mkdir()

    with pytest.raises(ValueError, match="run_dir must resolve under pipeline_runs_cs/"):
        resolve_pipeline_run_dir(
            "pipeline_runs_cs",
            pipeline_dir=pipeline_dir,
            root_dir=tmp_path,
        )


def test_resolve_pipeline_run_asset_path_validates_relative_lookup(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_demo"
    (run_dir / "meta").mkdir(parents=True)
    asset = run_dir / "meta" / "report.json"
    asset.write_text("{}", encoding="utf-8")

    assert resolve_pipeline_run_asset_path(run_path=run_dir, file="report.json", sub="meta") == asset

    with pytest.raises(ValueError, match="file must be relative"):
        resolve_pipeline_run_asset_path(run_path=run_dir, file=str(asset), sub=None)

    with pytest.raises(ValueError, match="Invalid file path"):
        resolve_pipeline_run_asset_path(run_path=run_dir, file="../secret.txt", sub=None)

    with pytest.raises(FileNotFoundError, match="File not found"):
        resolve_pipeline_run_asset_path(run_path=run_dir, file="missing.json", sub=None)


def test_build_pipeline_run_details_reads_meta_and_baseline_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "pipeline_runs_cs" / "run_demo"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)
    (run_dir / "SUMMARY.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    (meta_dir / "ui_context.json").write_text(json.dumps({"job_id": "demo"}), encoding="utf-8")
    (meta_dir / "run_metadata.json").write_text(json.dumps({"codename": "Bright Fox"}), encoding="utf-8")

    data_dir = tmp_path / "datasets" / "sp500"
    data_dir.mkdir(parents=True)
    (data_dir / "baseline_metrics.json").write_text(json.dumps({"sharpe": 0.8}), encoding="utf-8")
    (meta_dir / "backtest_config.json").write_text(
        json.dumps({"data_dir": str(data_dir)}),
        encoding="utf-8",
    )

    payload = build_pipeline_run_details(
        resolved_run_dir=run_dir,
        labels={"run_demo": "Focus"},
    )

    assert payload["path"] == str(run_dir)
    assert payload["name"] == "Bright Fox"
    assert payload["label"] == "Focus"
    assert payload["summary"] == {"schema_version": 1}
    assert payload["ui_context"] == {"job_id": "demo"}
    assert payload["meta"]["backtest_config"]["data_dir"] == str(data_dir)
    assert payload["baseline_metrics"] == {"sharpe": 0.8}


def test_update_pipeline_run_label_writes_and_clears_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_dir = tmp_path / "pipeline_runs_cs"
    pipeline_dir.mkdir()
    run_dir = pipeline_dir / "run_demo"
    run_dir.mkdir()

    monkeypatch.setattr(
        "alpha_evolve.dashboard.api.pipeline_runs.dashboard_helpers.ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "alpha_evolve.dashboard.api.pipeline_runs.dashboard_helpers.PIPELINE_DIR",
        pipeline_dir,
    )

    assert update_pipeline_run_label(path="pipeline_runs_cs/run_demo", label="Focus") == {"ok": True}
    labels_path = pipeline_dir / ".run_labels.json"
    assert json.loads(labels_path.read_text(encoding="utf-8")) == {"run_demo": "Focus"}

    assert update_pipeline_run_label(path="pipeline_runs_cs/run_demo", label="") == {"ok": True}
    assert json.loads(labels_path.read_text(encoding="utf-8")) == {}


def test_build_pipeline_runs_listing_uses_labels_and_codenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    newer = tmp_path / "run_newer"
    older = tmp_path / "run_older"
    for run_dir, codename in ((newer, "Silver Pine"), (older, "")):
        (run_dir / "meta").mkdir(parents=True)
        if codename:
            (run_dir / "meta" / "run_metadata.json").write_text(
                json.dumps({"codename": codename}),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "alpha_evolve.dashboard.api.pipeline_runs.find_pipeline_runs",
        lambda: [newer, older],
    )
    monkeypatch.setattr(
        "alpha_evolve.dashboard.api.pipeline_runs.dashboard_helpers.read_best_sharpe_from_run",
        lambda path: 1.2 if path == newer else None,
    )

    items = build_pipeline_runs_listing(limit=10, labels={"run_newer": "Priority"})

    assert items == [
        {
            "path": str(newer),
            "name": "Silver Pine",
            "label": "Priority",
            "sharpe_best": 1.2,
        },
        {
            "path": str(older),
            "name": "run_older",
            "label": None,
            "sharpe_best": None,
        },
    ]
