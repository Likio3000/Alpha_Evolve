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
        "checkpoint_gens": [30, 60, 90, 120],
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
            "gen_120": {
                "ensemble_portfolio_sharpe": {"mean": 1.35},
                "best_backtest_sharpe": {"mean": 1.15},
                "selected_avg_abs_corr": {"mean": 0.20},
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
                {
                    "from_gen": 90,
                    "to_gen": 120,
                    "metrics": [
                        {
                            "metric": "ensemble_portfolio_sharpe",
                            "mean_improvement": 0.05,
                            "ci95_mean_improvement": [0.01, 0.10],
                            "p_perm_one_sided_holm": 0.04,
                        },
                        {
                            "metric": "best_backtest_sharpe",
                            "mean_improvement": 0.05,
                            "ci95_mean_improvement": [0.005, 0.09],
                            "p_perm_one_sided_holm": 0.045,
                        },
                        {
                            "metric": "selected_avg_abs_corr",
                            "mean_improvement": corr_improvements[2],
                            "ci95_mean_improvement": [0.002, 0.03],
                            "p_perm_one_sided_holm": 0.045,
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
        json.dumps(_checkpoint_payload(corr_improvements=[0.04, 0.04, 0.02]), indent=2),
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
        json.dumps(_checkpoint_payload(corr_improvements=[0.04, -0.02, 0.02]), indent=2),
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
            "--checkpoint-direction-mode-correlation",
            "mean_positive",
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
    payload = _checkpoint_payload(corr_improvements=[0.02, 0.01, 0.01])
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


def test_scaling_goal_curve_block_rejects_three_point_direction_only_series() -> None:
    mod = _load_script_module("check_scaling_goal")
    payload = {
        "gen_030": {"ensemble_portfolio_sharpe": {"mean": 1.0}},
        "gen_060": {"ensemble_portfolio_sharpe": {"mean": 1.2}},
        "gen_090": {"ensemble_portfolio_sharpe": {"mean": 1.3}},
    }
    result = mod._evaluate_curve_block(
        payload,
        metrics_higher_is_better={"ensemble_portfolio_sharpe": True},
        alpha=0.05,
        tolerance=0.0,
    )
    metric = result["metrics"][0]
    assert metric["monotonic_pass"] is True
    assert metric["trend_pass"] is False
    assert metric["pass"] is False
    assert result["pass"] is False


def test_scaling_goal_slope_monte_carlo_pvalue_has_positive_floor() -> None:
    mod = _load_script_module("check_scaling_goal")
    x = json.loads(json.dumps(list(range(9))))
    y = json.loads(json.dumps(list(range(9))))
    p_val = mod._slope_perm_p_one_sided(
        mod.np.asarray(x, dtype=float),
        mod.np.asarray(y, dtype=float),
    )
    assert 0.0 < p_val < 1e-4


def test_scaling_goal_checker_practical_cross_regime_passes(tmp_path, monkeypatch) -> None:
    mod = _load_script_module("check_scaling_goal")
    out_path = tmp_path / "goal_check.json"

    def _practical_checkpoint(root: str) -> dict:
        payload = _checkpoint_payload(corr_improvements=[-0.0022, 0.0078, 0.0031])
        payload["root"] = root
        payload["checkpoint_gens"] = [30, 60, 90, 120, 200]
        payload["summary_by_generation"] = {
            "gen_030": {
                "ensemble_portfolio_sharpe": {"mean": 1.0},
                "best_backtest_sharpe": {"mean": 0.8},
                "selected_avg_abs_corr": {"mean": 0.2579},
            },
            "gen_060": {
                "ensemble_portfolio_sharpe": {"mean": 1.1},
                "best_backtest_sharpe": {"mean": 0.82},
                "selected_avg_abs_corr": {"mean": 0.2601},
            },
            "gen_090": {
                "ensemble_portfolio_sharpe": {"mean": 1.2},
                "best_backtest_sharpe": {"mean": 0.84},
                "selected_avg_abs_corr": {"mean": 0.2523},
            },
            "gen_120": {
                "ensemble_portfolio_sharpe": {"mean": 1.27},
                "best_backtest_sharpe": {"mean": 0.85},
                "selected_avg_abs_corr": {"mean": 0.2492},
            },
            "gen_200": {
                "ensemble_portfolio_sharpe": {"mean": 1.33},
                "best_backtest_sharpe": {"mean": 0.855},
                "selected_avg_abs_corr": {"mean": 0.2375},
            },
        }
        best_bt_rows = [
            metric
            for pair in payload["pairwise_scientific"]["pairs"]
            for metric in pair["metrics"]
            if metric["metric"] == "best_backtest_sharpe"
        ]
        for row in best_bt_rows:
            row["ci95_mean_improvement"] = [-0.01, 0.03]
            row["p_perm_one_sided_holm"] = 0.2
        corr_rows = [
            metric
            for pair in payload["pairwise_scientific"]["pairs"]
            for metric in pair["metrics"]
            if metric["metric"] == "selected_avg_abs_corr"
        ]
        corr_rows[0]["ci95_mean_improvement"] = [-0.016, 0.012]
        corr_rows[0]["p_perm_one_sided_holm"] = 0.6
        corr_rows[1]["ci95_mean_improvement"] = [0.001, 0.015]
        corr_rows[1]["p_perm_one_sided_holm"] = 0.04
        corr_rows[2]["ci95_mean_improvement"] = [-0.003, 0.009]
        corr_rows[2]["p_perm_one_sided_holm"] = 0.25
        return payload

    def _scientific(root: str) -> dict:
        payload = _scientific_payload()
        payload["control_root"] = f"{root}/analysis_g200_vs_g60/control_g60_checkpoint"
        payload["treatment_root"] = f"{root}/analysis_g200_vs_g60/treatment_g200_final"
        return payload

    root_a = str((tmp_path / "campaign_a").resolve())
    root_b = str((tmp_path / "campaign_b").resolve())
    ckpt_a = tmp_path / "regime_a_checkpoint.json"
    ckpt_b = tmp_path / "regime_b_checkpoint.json"
    sci_a = tmp_path / "regime_a_scientific.json"
    sci_b = tmp_path / "regime_b_scientific.json"
    ckpt_a.write_text(json.dumps(_practical_checkpoint(root_a), indent=2), encoding="utf-8")
    ckpt_b.write_text(json.dumps(_practical_checkpoint(root_b), indent=2), encoding="utf-8")
    sci_a.write_text(json.dumps(_scientific(root_a), indent=2), encoding="utf-8")
    sci_b.write_text(json.dumps(_scientific(root_b), indent=2), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_scaling_goal.py",
            "--checkpoint-summary-json",
            str(ckpt_a),
            "--checkpoint-summary-json",
            str(ckpt_b),
            "--scientific-json",
            str(sci_a),
            "--scientific-json",
            str(sci_b),
            "--min-regimes",
            "2",
            "--out",
            str(out_path),
        ],
    )
    rc = mod.main()
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_thresholds"]["direction_mode_correlation"] == "ci_nonnegative"
    assert payload["checkpoint_thresholds"]["significance_aggregation_positive"] == "pooled"
    assert payload["checkpoint_thresholds"]["tolerance"] == 0.0023
    assert payload["distinct_regime_count"] == 2
    assert payload["cross_regime_pass"] is True
    assert payload["overall_goal_pass"] is True


def test_scaling_goal_checker_cross_regime_fails_when_evidence_missing(tmp_path, monkeypatch) -> None:
    mod = _load_script_module("check_scaling_goal")
    out_path = tmp_path / "goal_check.json"
    root_a = str((tmp_path / "campaign_a").resolve())
    root_b = str((tmp_path / "campaign_b").resolve())
    payload_a = _checkpoint_payload(corr_improvements=[0.04, 0.04, 0.02])
    payload_b = _checkpoint_payload(corr_improvements=[0.04, 0.04, 0.02])
    payload_a["root"] = root_a
    payload_b["root"] = root_b
    ckpt_a = tmp_path / "regime_a_checkpoint.json"
    ckpt_b = tmp_path / "regime_b_checkpoint.json"
    sci_a = tmp_path / "regime_a_scientific.json"
    ckpt_a.write_text(json.dumps(payload_a, indent=2), encoding="utf-8")
    ckpt_b.write_text(json.dumps(payload_b, indent=2), encoding="utf-8")
    scientific = _scientific_payload()
    scientific["control_root"] = f"{root_a}/analysis_g200_vs_g60/control_g60_checkpoint"
    scientific["treatment_root"] = f"{root_a}/analysis_g200_vs_g60/treatment_g200_final"
    sci_a.write_text(json.dumps(scientific, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_scaling_goal.py",
            "--checkpoint-summary-json",
            str(ckpt_a),
            "--checkpoint-summary-json",
            str(ckpt_b),
            "--scientific-json",
            str(sci_a),
            "--min-regimes",
            "2",
            "--out",
            str(out_path),
        ],
    )
    rc = mod.main()
    assert rc == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["distinct_regime_count"] == 1
    assert payload["cross_regime_pass"] is False
    assert payload["missing_family_signatures"] == [root_b]
