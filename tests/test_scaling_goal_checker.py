from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(stem: str):
    path = ROOT / "scripts" / f"{stem}.py"
    module_name = f"test_script_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_payload(*, corr_improvements: list[float]) -> dict:
    return {
        "schema_version": 1,
        "checkpoint_gens": [30, 60, 90],
        "summary_by_generation": {
            "gen_030": {
                "ensemble_portfolio_sharpe": {"mean": 1.0},
                "best_backtest_sharpe": {"mean": 0.8},
                "selected_avg_abs_corr": {"mean": 0.30},
            },
            "gen_060": {
                "ensemble_portfolio_sharpe": {"mean": 1.2},
                "best_backtest_sharpe": {"mean": 1.0},
                "selected_avg_abs_corr": {"mean": 0.26},
            },
            "gen_090": {
                "ensemble_portfolio_sharpe": {"mean": 1.3},
                "best_backtest_sharpe": {"mean": 1.1},
                "selected_avg_abs_corr": {"mean": 0.22},
            },
        },
        "pairwise_scientific": {
            "schema_version": 1,
            "pairs": [
                {
                    "from_gen": 30,
                    "to_gen": 60,
                    "metrics": [
                        {
                            "metric": "ensemble_portfolio_sharpe",
                            "mean_improvement": 0.2,
                            "ci95_mean_improvement": [0.05, 0.35],
                            "p_perm_one_sided_holm": 0.01,
                        },
                        {
                            "metric": "best_backtest_sharpe",
                            "mean_improvement": 0.2,
                            "ci95_mean_improvement": [0.05, 0.30],
                            "p_perm_one_sided_holm": 0.02,
                        },
                        {
                            "metric": "selected_avg_abs_corr",
                            "mean_improvement": corr_improvements[0],
                            "ci95_mean_improvement": [0.01, 0.08],
                            "p_perm_one_sided_holm": 0.02,
                        },
                    ],
                },
                {
                    "from_gen": 60,
                    "to_gen": 90,
                    "metrics": [
                        {
                            "metric": "ensemble_portfolio_sharpe",
                            "mean_improvement": 0.1,
                            "ci95_mean_improvement": [0.02, 0.20],
                            "p_perm_one_sided_holm": 0.03,
                        },
                        {
                            "metric": "best_backtest_sharpe",
                            "mean_improvement": 0.1,
                            "ci95_mean_improvement": [0.01, 0.18],
                            "p_perm_one_sided_holm": 0.04,
                        },
                        {
                            "metric": "selected_avg_abs_corr",
                            "mean_improvement": corr_improvements[1],
                            "ci95_mean_improvement": [0.005, 0.05],
                            "p_perm_one_sided_holm": 0.04,
                        },
                    ],
                },
            ],
        },
    }


def _scientific_payload() -> dict:
    return {
        "schema_version": 1,
        "results": [
            {
                "metric": "ensemble_sharpe",
                "mean_improvement": 0.12,
                "ci95_mean_improvement": [0.04, 0.20],
                "p_perm_one_sided_holm": 0.01,
            },
            {
                "metric": "ensemble_annret",
                "mean_improvement": 0.02,
                "ci95_mean_improvement": [0.005, 0.03],
                "p_perm_one_sided_holm": 0.02,
            },
            {
                "metric": "pair_mean_abs_corr",
                "mean_improvement": 0.01,
                "ci95_mean_improvement": [0.001, 0.02],
                "p_perm_one_sided_holm": 0.03,
            },
        ],
    }


def test_scaling_goal_checker_passes_for_clean_monotonic_case(tmp_path, monkeypatch) -> None:
    mod = _load_script_module("check_scaling_goal")
    ckpt_path = tmp_path / "checkpoint_summary.json"
    sci_path = tmp_path / "scientific.json"
    out_path = tmp_path / "goal_check.json"
    ckpt_path.write_text(
        json.dumps(_checkpoint_payload(corr_improvements=[0.04, 0.04]), indent=2),
        encoding="utf-8",
    )
    sci_path.write_text(json.dumps(_scientific_payload(), indent=2), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_scaling_goal.py",
            "--checkpoint-summary-json",
            str(ckpt_path),
            "--scientific-json",
            str(sci_path),
            "--out",
            str(out_path),
        ],
    )
    rc = mod.main()
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["overall_goal_pass"] is True
    assert payload["checkpoint_pass"] is True
    assert payload["scientific_pass"] is True


def test_scaling_goal_checker_fails_when_corr_step_regresses(tmp_path, monkeypatch) -> None:
    mod = _load_script_module("check_scaling_goal")
    ckpt_path = tmp_path / "checkpoint_summary.json"
    out_path = tmp_path / "goal_check.json"
    # Second step has negative oriented improvement in correlation.
    ckpt_path.write_text(
        json.dumps(_checkpoint_payload(corr_improvements=[0.04, -0.02]), indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_scaling_goal.py",
            "--checkpoint-summary-json",
            str(ckpt_path),
            "--required-scientific-positive",
            "",
            "--required-scientific-corr",
            "",
            "--out",
            str(out_path),
        ],
    )
    rc = mod.main()
    assert rc == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["overall_goal_pass"] is False
    assert payload["checkpoint_pass"] is False


def test_scaling_goal_checker_relaxed_sig_fraction_for_corr(tmp_path, monkeypatch) -> None:
    mod = _load_script_module("check_scaling_goal")
    ckpt_path = tmp_path / "checkpoint_summary.json"
    out_path = tmp_path / "goal_check.json"
    # Direction is positive in both correlation steps, but significance is absent.
    payload = _checkpoint_payload(corr_improvements=[0.02, 0.01])
    for pair in payload["pairwise_scientific"]["pairs"]:
        for m in pair["metrics"]:
            if m["metric"] == "selected_avg_abs_corr":
                m["ci95_mean_improvement"] = [-0.01, 0.05]
                m["p_perm_one_sided_holm"] = 0.4
    ckpt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_scaling_goal.py",
            "--checkpoint-summary-json",
            str(ckpt_path),
            "--required-scientific-positive",
            "",
            "--required-scientific-corr",
            "",
            "--checkpoint-min-significant-fraction-correlation",
            "0.0",
            "--out",
            str(out_path),
        ],
    )
    rc = mod.main()
    assert rc == 0
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["checkpoint_pass"] is True
