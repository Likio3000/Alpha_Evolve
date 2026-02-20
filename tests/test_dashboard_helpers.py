from __future__ import annotations

from alpha_evolve.dashboard.api.helpers import build_pipeline_args


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
