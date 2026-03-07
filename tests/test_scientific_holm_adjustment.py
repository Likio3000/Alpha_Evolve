from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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


def test_holm_adjustment_matches_expected_values_across_scripts() -> None:
    expected = [0.03, 0.06, 0.06]
    inputs = [0.01, 0.04, 0.03]
    for stem in (
        "scientific_compare",
        "benchmark_sp500",
        "analyze_completed_runs",
        "aggregate_parallel_benchmarks",
    ):
        mod = _load_script_module(stem)
        adjusted = mod._holm_bonferroni_adjust(inputs)
        assert len(adjusted) == len(expected)
        for got, exp in zip(adjusted, expected):
            assert got == exp


def test_checkpoint_pairwise_report_uses_holm_adjusted_scientific_pass() -> None:
    rows: list[dict[str, float | int]] = []
    for seed in range(5):
        rows.append(
            {
                "seed": seed,
                "generation": 1,
                "best_backtest_sharpe": 1.0,
                "ensemble_portfolio_sharpe": 0.0,
                "selected_avg_abs_corr": 0.5,
            }
        )
        rows.append(
            {
                "seed": seed,
                "generation": 2,
                "best_backtest_sharpe": 2.0,
                "ensemble_portfolio_sharpe": 0.0,
                "selected_avg_abs_corr": 0.5,
            }
        )

    for stem in (
        "benchmark_sp500",
        "analyze_completed_runs",
        "aggregate_parallel_benchmarks",
    ):
        mod = _load_script_module(stem)
        if stem == "aggregate_parallel_benchmarks":
            payload = mod._checkpoint_pairwise_report(pd.DataFrame(rows))
        else:
            payload = mod._checkpoint_pairwise_report(rows)
        pair = payload["pairs"][0]
        metric = next(m for m in pair["metrics"] if m["metric"] == "best_backtest_sharpe")
        assert metric["scientific_pass_unadjusted"] is True
        assert metric["scientific_pass"] is False
        assert metric["p_perm_one_sided_holm"] > 0.05


def test_checkpoint_pairwise_report_holm_adjusts_across_all_pairs() -> None:
    rows: list[dict[str, float | int]] = []
    for seed in range(5):
        rows.extend(
            [
                {"seed": seed, "generation": 1, "best_backtest_sharpe": 1.0},
                {"seed": seed, "generation": 2, "best_backtest_sharpe": 2.0},
                {"seed": seed, "generation": 3, "best_backtest_sharpe": 3.0},
            ]
        )

    for stem in (
        "benchmark_sp500",
        "analyze_completed_runs",
        "aggregate_parallel_benchmarks",
    ):
        mod = _load_script_module(stem)
        if stem == "aggregate_parallel_benchmarks":
            payload = mod._checkpoint_pairwise_report(pd.DataFrame(rows))
        else:
            payload = mod._checkpoint_pairwise_report(rows)
        metrics = [pair["metrics"][0] for pair in payload["pairs"]]
        assert all(metric["scientific_pass_unadjusted"] is True for metric in metrics)
        assert all(metric["scientific_pass"] is False for metric in metrics)
        assert all(metric["p_perm_one_sided_holm"] > 0.05 for metric in metrics)


def test_permutation_monte_carlo_pvalues_have_positive_floor() -> None:
    vals = np.ones(21, dtype=float)
    for stem in (
        "scientific_compare",
        "benchmark_sp500",
        "analyze_completed_runs",
        "aggregate_parallel_benchmarks",
    ):
        mod = _load_script_module(stem)
        p_one, p_two = mod._paired_sign_flip_pvalues(vals)
        assert 0.0 < p_one < 1e-4
        assert 0.0 < p_two < 1e-4


def test_scientific_compare_rejects_duplicate_seed_dirs(tmp_path) -> None:
    mod = _load_script_module("scientific_compare")
    root = tmp_path / "dup_root"

    def _write_run(run_name: str) -> None:
        run_dir = root / run_name
        bt_dir = run_dir / "backtest_portfolio_csvs"
        bt_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "SUMMARY.json").write_text(
            json.dumps({"best_metrics": {"Sharpe": 1.0}}), encoding="utf-8"
        )
        with (bt_dir / "backtest_summary_ensemble.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["Sharpe", "AnnReturn", "MaxDD"], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerow({"Sharpe": 0.0, "AnnReturn": 0.0, "MaxDD": 0.1})
        with (bt_dir / "return_corr_matrix.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["", "a", "b"])
            writer.writerow(["a", "1.0", "0.2"])
            writer.writerow(["b", "0.2", "1.0"])
        (bt_dir / "ensemble_selection.json").write_text(
            json.dumps({"members": ["a", "b"]}), encoding="utf-8"
        )

    _write_run("run_seed1_a")
    _write_run("run_seed1_b")

    with pytest.raises(ValueError, match="Duplicate seed 1"):
        mod._collect_group(root)


def test_scientific_compare_main_applies_holm_adjustment(tmp_path, monkeypatch) -> None:
    mod = _load_script_module("scientific_compare")
    control_root = tmp_path / "control"
    treatment_root = tmp_path / "treatment"
    out_path = tmp_path / "scientific.json"

    def _write_run(root: Path, seed: int, best_sharpe: float) -> None:
        run_dir = root / f"run_seed{seed}"
        bt_dir = run_dir / "backtest_portfolio_csvs"
        bt_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "SUMMARY.json").write_text(
            json.dumps({"best_metrics": {"Sharpe": best_sharpe}}), encoding="utf-8"
        )

        with (bt_dir / "backtest_summary_ensemble.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["Sharpe", "AnnReturn", "MaxDD"], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerow({"Sharpe": 0.0, "AnnReturn": 0.0, "MaxDD": 0.1})

        with (bt_dir / "return_corr_matrix.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["", "a", "b"])
            writer.writerow(["a", "1.0", "0.2"])
            writer.writerow(["b", "0.2", "1.0"])

        (bt_dir / "ensemble_selection.json").write_text(
            json.dumps({"members": ["a", "b"]}), encoding="utf-8"
        )

    for seed in range(5):
        _write_run(control_root, seed, best_sharpe=0.0)
        _write_run(treatment_root, seed, best_sharpe=1.0)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scientific_compare.py",
            "--control-root",
            str(control_root),
            "--treatment-root",
            str(treatment_root),
            "--out",
            str(out_path),
            "--bootstrap",
            "1000",
            "--seed",
            "123",
        ],
    )
    rc = mod.main()
    assert rc == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    best = next(x for x in payload["results"] if x["metric"] == "best_sharpe")
    assert best["scientific_pass_unadjusted"] is True
    assert best["scientific_pass"] is False
    assert best["p_perm_one_sided_holm"] > 0.05
