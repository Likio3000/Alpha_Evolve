from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from alpha_evolve.dashboard.api.ml_lab_runs import (
    format_ml_lab_path_for_ui,
    plan_ml_lab_run,
    resolve_ml_lab_run_dir,
)


def test_plan_ml_lab_run_creates_expected_directory(tmp_path: Path) -> None:
    plan = plan_ml_lab_run(
        payload={"seed": 7},
        dataset="sp500_small",
        cfg_path=None,
        now=datetime(2026, 3, 8, 16, 5, 4),
        runs_dir=tmp_path,
        codename_factory=lambda: "Ava-Smith",
    )

    assert plan.run_stamp == "20260308_160504"
    assert plan.seed_label == 7
    assert plan.dataset_label == "sp500_small"
    assert plan.run_dir == tmp_path / "run_ml_Ava-Smith_seed7_sp500_small_20260308_160504"
    assert plan.run_dir.exists()
    assert plan.spec_path == plan.run_dir / "ml_spec.json"


def test_plan_ml_lab_run_uses_config_stem_and_retries_existing_directory(tmp_path: Path) -> None:
    existing = tmp_path / "run_ml_Ava-Smith_seed42_my_config_20260308_160504"
    existing.mkdir(parents=True)

    plan = plan_ml_lab_run(
        payload={},
        dataset="",
        cfg_path="/tmp/my config.toml",
        now=datetime(2026, 3, 8, 16, 5, 4),
        runs_dir=tmp_path,
        codename_factory=lambda: "Ava-Smith",
    )

    assert plan.seed_label == 42
    assert plan.dataset_label == "my_config"
    assert plan.run_dir == tmp_path / "run_ml_Ava-Smith-1_seed42_my_config_20260308_160504"
    assert plan.run_dir.exists()


def test_resolve_ml_lab_run_dir_accepts_relative_paths_under_runs_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_demo"
    run_dir.mkdir()

    resolved = resolve_ml_lab_run_dir("run_demo", runs_dir=tmp_path, root_dir=tmp_path.parent)

    assert resolved == run_dir.resolve()


def test_resolve_ml_lab_run_dir_rejects_paths_outside_runs_dir(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="run_dir must resolve under pipeline_runs_cs/ml_runs"):
        resolve_ml_lab_run_dir(str(outside), runs_dir=tmp_path, root_dir=tmp_path.parent)


def test_format_ml_lab_path_for_ui_prefers_root_relative(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    pipeline_dir = root_dir / "pipeline_runs_cs"
    run_dir = pipeline_dir / "ml_runs" / "run_demo"
    run_dir.mkdir(parents=True)

    assert (
        format_ml_lab_path_for_ui(run_dir, root_dir=root_dir, pipeline_dir=pipeline_dir)
        == "pipeline_runs_cs/ml_runs/run_demo"
    )
