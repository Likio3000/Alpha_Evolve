#!/usr/bin/env python3
"""Scientific paired comparison for two sets of pipeline runs.

Compares treatment vs control by matching `seed<INT>` in run directory names,
then reports paired improvement statistics with bootstrap confidence intervals,
exact sign-test p-values, and paired sign-flip permutation p-values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SEED_RE = re.compile(r"seed(\d+)")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool


DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("best_sharpe", True),
    MetricSpec("ensemble_sharpe", True),
    MetricSpec("ensemble_annret", True),
    MetricSpec("ensemble_maxdd", False),
    MetricSpec("pair_mean_abs_corr", False),
    MetricSpec("ensemble_mean_abs_corr", False),
    MetricSpec("dup_pairs_ge_0_999", False),
)


def _parse_seed(run_name: str) -> int | None:
    m = SEED_RE.search(run_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        return 0.0
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denom) or denom < 1e-12:
        return 0.0
    c = float(np.dot(a, b) / denom)
    if not np.isfinite(c):
        return 0.0
    return float(max(-1.0, min(1.0, c)))


def _collect_run_metrics(run_dir: Path) -> dict[str, float]:
    summary = json.loads((run_dir / "SUMMARY.json").read_text(encoding="utf-8"))
    bt_dir = run_dir / "backtest_portfolio_csvs"

    best_sharpe = float(summary.get("best_metrics", {}).get("Sharpe", np.nan))

    ens_csv = bt_dir / "backtest_summary_ensemble.csv"
    ens = pd.read_csv(ens_csv) if ens_csv.exists() else pd.DataFrame()
    ensemble_sharpe = float(ens.iloc[0]["Sharpe"]) if len(ens) and "Sharpe" in ens else np.nan
    ensemble_annret = float(ens.iloc[0]["AnnReturn"]) if len(ens) and "AnnReturn" in ens else np.nan
    ensemble_maxdd = float(ens.iloc[0]["MaxDD"]) if len(ens) and "MaxDD" in ens else np.nan

    cm_path = bt_dir / "return_corr_matrix.csv"
    pair_mean_abs_corr = np.nan
    pair_max_abs_corr = np.nan
    dup_pairs = np.nan
    ensemble_mean_abs_corr = np.nan
    ensemble_max_abs_corr = np.nan
    if cm_path.exists():
        cm = pd.read_csv(cm_path, index_col=0)
        vals = []
        for i in range(len(cm)):
            for j in range(i + 1, len(cm)):
                vals.append(abs(float(cm.iloc[i, j])))
        if vals:
            arr = np.asarray(vals, dtype=float)
            pair_mean_abs_corr = float(np.mean(arr))
            pair_max_abs_corr = float(np.max(arr))
            dup_pairs = float(np.sum(arr >= 0.999))

        ens_sel_path = bt_dir / "ensemble_selection.json"
        if ens_sel_path.exists():
            members = list(json.loads(ens_sel_path.read_text(encoding="utf-8")).get("members") or [])
            members = [m for m in members if m in cm.index]
            if len(members) >= 2:
                sub = cm.loc[members, members].to_numpy(dtype=float)
                vals2 = []
                for i in range(len(sub)):
                    for j in range(i + 1, len(sub)):
                        vals2.append(abs(float(sub[i, j])))
                if vals2:
                    arr2 = np.asarray(vals2, dtype=float)
                    ensemble_mean_abs_corr = float(np.mean(arr2))
                    ensemble_max_abs_corr = float(np.max(arr2))

    return {
        "best_sharpe": best_sharpe,
        "ensemble_sharpe": ensemble_sharpe,
        "ensemble_annret": ensemble_annret,
        "ensemble_maxdd": ensemble_maxdd,
        "pair_mean_abs_corr": pair_mean_abs_corr,
        "pair_max_abs_corr": pair_max_abs_corr,
        "ensemble_mean_abs_corr": ensemble_mean_abs_corr,
        "ensemble_max_abs_corr": ensemble_max_abs_corr,
        "dup_pairs_ge_0_999": dup_pairs,
    }


def _collect_group(root: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for run_dir in sorted(root.glob("run_*")):
        seed = _parse_seed(run_dir.name)
        if seed is None:
            continue
        try:
            out[seed] = _collect_run_metrics(run_dir)
        except Exception:
            continue
    return out


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        v = float(values[0])
        return (v, v)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = np.mean(values[idx], axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _sign_test_p_one_sided_improvement(values: np.ndarray) -> float:
    # H1: improvement > 0
    nz = values[values != 0]
    n = int(nz.size)
    if n == 0:
        return 1.0
    k = int(np.sum(nz > 0))
    numer = 0
    denom = 2**n
    for i in range(k, n + 1):
        numer += math.comb(n, i)
    return float(numer / denom)


def _paired_sign_flip_pvalues(values: np.ndarray) -> tuple[float, float]:
    vals = values[np.isfinite(values)]
    n = int(vals.size)
    if n == 0:
        return 1.0, 1.0
    obs = float(np.mean(vals))
    if n <= 20:
        all_means = []
        total = 1 << n
        for mask in range(total):
            signs = np.ones(n, dtype=float)
            for i in range(n):
                if (mask >> i) & 1:
                    signs[i] = -1.0
            all_means.append(float(np.mean(vals * signs)))
        arr = np.asarray(all_means, dtype=float)
        p_one = float(np.mean(arr >= obs))
        p_two = float(np.mean(np.abs(arr) >= abs(obs)))
        return p_one, p_two

    rng = np.random.default_rng(12345)
    n_perm = 200000
    signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(n_perm, n))
    means = np.mean(signs * vals[None, :], axis=1)
    p_one = float(np.mean(means >= obs))
    p_two = float(np.mean(np.abs(means) >= abs(obs)))
    return p_one, p_two


def _evaluate_metric(
    metric: MetricSpec,
    control_vals: np.ndarray,
    treatment_vals: np.ndarray,
    *,
    alpha: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    if metric.higher_is_better:
        improvements = treatment_vals - control_vals
    else:
        improvements = control_vals - treatment_vals

    imp = improvements[np.isfinite(improvements)]
    c = control_vals[np.isfinite(control_vals)]
    t = treatment_vals[np.isfinite(treatment_vals)]
    rng = np.random.default_rng(seed)
    ci_lo, ci_hi = _bootstrap_mean_ci(imp, n_boot=n_boot, alpha=alpha, rng=rng)
    p_sign = _sign_test_p_one_sided_improvement(imp)
    p_perm_one, p_perm_two = _paired_sign_flip_pvalues(imp)
    mean_imp = float(np.mean(imp)) if imp.size else float("nan")
    median_imp = float(np.median(imp)) if imp.size else float("nan")
    std_imp = float(np.std(imp, ddof=1)) if imp.size > 1 else float("nan")
    effect = float(mean_imp / std_imp) if np.isfinite(std_imp) and std_imp > 1e-12 else float("nan")
    improved_frac = float(np.mean(imp > 0)) if imp.size else float("nan")
    eps = 1e-12
    positives = int(np.sum(imp > eps))
    negatives = int(np.sum(imp < -eps))
    zeros = int(imp.size - positives - negatives)
    non_negative_fraction = float(np.mean(imp >= -eps)) if imp.size else float("nan")

    return {
        "metric": metric.name,
        "higher_is_better": metric.higher_is_better,
        "n": int(imp.size),
        "control_mean": float(np.mean(c)) if c.size else float("nan"),
        "treatment_mean": float(np.mean(t)) if t.size else float("nan"),
        "mean_improvement": mean_imp,
        "median_improvement": median_imp,
        "std_improvement": std_imp,
        "effect_size_d": effect,
        "improved_fraction": improved_frac,
        "positives": positives,
        "zeros": zeros,
        "negatives": negatives,
        "non_negative_fraction": non_negative_fraction,
        "monotonic_non_decreasing_all": bool(negatives == 0),
        "scientific_pass_non_decrease": bool(negatives == 0 and positives > 0),
        "ci95_mean_improvement": [ci_lo, ci_hi],
        "p_sign_one_sided": p_sign,
        "p_perm_one_sided": p_perm_one,
        "p_perm_two_sided": p_perm_two,
        "scientific_pass": bool(np.isfinite(ci_lo) and ci_lo > 0 and p_perm_one <= alpha),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired scientific comparison of run groups")
    p.add_argument("--control-root", required=True, help="Directory containing control run_* folders")
    p.add_argument("--treatment-root", required=True, help="Directory containing treatment run_* folders")
    p.add_argument("--out", default=None, help="Optional output JSON path")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    p.add_argument("--bootstrap", type=int, default=50000, help="Bootstrap resamples for CI")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for bootstrap")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    control_root = Path(args.control_root).resolve()
    treatment_root = Path(args.treatment_root).resolve()
    control = _collect_group(control_root)
    treatment = _collect_group(treatment_root)
    common_seeds = sorted(set(control.keys()).intersection(treatment.keys()))
    if not common_seeds:
        raise SystemExit("No overlapping seeds between control and treatment groups.")

    results: list[dict[str, Any]] = []
    for spec in DEFAULT_METRICS:
        c_vals = []
        t_vals = []
        for s in common_seeds:
            c = control[s].get(spec.name, np.nan)
            t = treatment[s].get(spec.name, np.nan)
            if pd.notna(c) and pd.notna(t):
                c_vals.append(float(c))
                t_vals.append(float(t))
        if not c_vals:
            continue
        res = _evaluate_metric(
            spec,
            np.asarray(c_vals, dtype=float),
            np.asarray(t_vals, dtype=float),
            alpha=float(args.alpha),
            n_boot=int(args.bootstrap),
            seed=int(args.seed),
        )
        results.append(res)

    payload = {
        "schema_version": 1,
        "control_root": str(control_root),
        "treatment_root": str(treatment_root),
        "common_seeds": common_seeds,
        "alpha": float(args.alpha),
        "bootstrap": int(args.bootstrap),
        "results": results,
    }

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Scientific comparison (positive mean_improvement favors treatment):")
    for r in results:
        ci = r["ci95_mean_improvement"]
        print(
            f"- {r['metric']}: n={r['n']} "
            f"mean_imp={r['mean_improvement']:.6f} "
            f"CI95=[{ci[0]:.6f},{ci[1]:.6f}] "
            f"p_perm_one={r['p_perm_one_sided']:.4f} "
            f"pass={r['scientific_pass']}"
        )

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
