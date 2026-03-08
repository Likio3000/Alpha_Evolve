from __future__ import annotations

from pathlib import Path

from alpha_evolve.dashboard.api.helpers import ROOT, build_pipeline_args, resolve_config_path
from alpha_evolve.utils.run_artifacts import resolve_latest_run_dir, select_backtest_summary_csv


def test_build_pipeline_args_respects_generations_override() -> None:
    args = build_pipeline_args(
        {"generations": 5, "overrides": {"generations": 11}},
        include_runner=False,
    )
    assert args[0] == "11"


def test_build_pipeline_args_maps_bt_top_alias_and_false_bools() -> None:
    args = build_pipeline_args(
        {
            "generations": 3,
            "overrides": {
                "bt_top": 7,
                "use_train_val_splits": False,
                "dry_run": False,
            },
        },
        include_runner=False,
    )
    assert "--top_to_backtest" in args
    idx = args.index("--top_to_backtest")
    assert args[idx + 1] == "7"
    assert "--no-use_train_val_splits" in args
    # dry_run does not support --no-dry_run; false should be omitted.
    assert "--no-dry_run" not in args
    assert "--dry_run" not in args


def test_build_pipeline_args_emits_no_flag_for_true_default_bools() -> None:
    args = build_pipeline_args(
        {"generations": 2, "overrides": {"ensemble_relax_corr": False}},
        include_runner=False,
    )
    assert "--no-ensemble_relax_corr" in args


def test_resolve_config_path_prefers_repo_relative_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_config_path("configs/sp500.toml")
    assert resolved == (ROOT / "configs" / "sp500.toml").resolve()


def test_select_backtest_summary_csv_prefers_highest_numeric_topn(tmp_path: Path) -> None:
    bt_dir = tmp_path / "backtest_portfolio_csvs"
    bt_dir.mkdir()
    (bt_dir / "backtest_summary_top2.csv").write_text("Sharpe\n1.0\n", encoding="utf-8")
    (bt_dir / "backtest_summary_top10.csv").write_text("Sharpe\n2.0\n", encoding="utf-8")

    selected = select_backtest_summary_csv(bt_dir)
    assert selected == (bt_dir / "backtest_summary_top10.csv")


def test_resolve_latest_run_dir_falls_back_when_latest_is_stale(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline_runs_cs"
    pipeline_dir.mkdir()
    older = pipeline_dir / "run_older"
    older.mkdir()
    newest = pipeline_dir / "run_newest"
    newest.mkdir()
    (pipeline_dir / "LATEST").write_text("run_missing", encoding="utf-8")

    resolved = resolve_latest_run_dir(pipeline_dir, project_root=tmp_path)
    assert resolved == newest
