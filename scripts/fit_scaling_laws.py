#!/usr/bin/env python3
"""Fit compute-scaling laws from matched run groups.

Inputs are run roots (containing run_* directories) keyed by compute level.
The script matches seeds across all compute levels, computes per-seed metric
improvements versus a baseline compute, then fits:
1) log-linear: y = a + b*log(c)
2) power-law:  y = a + b*c^k
3) piecewise log-linear: y = a + b1*log(c) + b2*max(0, log(c)-knot)

It also performs leave-one-compute-out predictive checks and bootstrap
uncertainty over seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SEED_RE = re.compile(r"seed(\d+)")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool


DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("ensemble_sharpe", True),
    MetricSpec("ensemble_annret", True),
    MetricSpec("pair_mean_abs_corr", False),
)


def _parse_seed(run_name: str) -> int | None:
    m = SEED_RE.search(run_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _collect_run_metrics(run_dir: Path) -> dict[str, float]:
    summary = json.loads((run_dir / "SUMMARY.json").read_text(encoding="utf-8"))
    bt_dir = run_dir / "backtest_portfolio_csvs"

    ens_csv = bt_dir / "backtest_summary_ensemble.csv"
    ens = pd.read_csv(ens_csv) if ens_csv.exists() else pd.DataFrame()
    ensemble_sharpe = float(ens.iloc[0]["Sharpe"]) if len(ens) and "Sharpe" in ens else np.nan
    ensemble_annret = float(ens.iloc[0]["AnnReturn"]) if len(ens) and "AnnReturn" in ens else np.nan
    ensemble_maxdd = float(ens.iloc[0]["MaxDD"]) if len(ens) and "MaxDD" in ens else np.nan

    best_sharpe = float(summary.get("best_metrics", {}).get("Sharpe", np.nan))

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


def _fit_linear(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    sse = float(np.sum((y - yhat) ** 2))
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "sse": sse}


def _fit_log_linear(c: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.log(c)
    fit = _fit_linear(x, y)
    return {
        "model": "log_linear",
        "intercept": fit["intercept"],
        "slope_log_compute": fit["slope"],
        "sse": fit["sse"],
    }


def _fit_power_grid(c: np.ndarray, y: np.ndarray, *, k_min: float = -1.0, k_max: float = 2.0, n_k: int = 301) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    ks = np.linspace(k_min, k_max, n_k)
    for k in ks:
        z = np.power(c, k)
        fit = _fit_linear(z, y)
        cand = {"k": float(k), "intercept": fit["intercept"], "coef": fit["slope"], "sse": fit["sse"]}
        if best is None or cand["sse"] < best["sse"]:
            best = cand
    assert best is not None
    return {
        "model": "power_law",
        "intercept": best["intercept"],
        "coef": best["coef"],
        "exponent_k": best["k"],
        "sse": best["sse"],
    }


def _fit_piecewise_log(c: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.log(c)
    if x.size < 4:
        base = _fit_log_linear(c, y)
        return {
            "model": "piecewise_log_linear",
            "knot_compute": None,
            "slope_left": base["slope_log_compute"],
            "slope_right": base["slope_log_compute"],
            "intercept": base["intercept"],
            "sse": base["sse"],
        }

    best: dict[str, Any] | None = None
    for knot_idx in range(1, x.size - 1):
        knot = float(x[knot_idx])
        hinge = np.maximum(0.0, x - knot)
        X = np.column_stack([np.ones_like(x), x, hinge])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        sse = float(np.sum((y - yhat) ** 2))
        b1 = float(beta[1])
        b2 = float(beta[2])
        cand = {
            "knot_idx": knot_idx,
            "knot_log_compute": knot,
            "knot_compute": float(np.exp(knot)),
            "intercept": float(beta[0]),
            "slope_left": b1,
            "slope_right": b1 + b2,
            "sse": sse,
        }
        if best is None or cand["sse"] < best["sse"]:
            best = cand
    assert best is not None
    return {"model": "piecewise_log_linear", **best}


def _predict_log_linear(fit: dict[str, Any], c: float) -> float:
    return float(fit["intercept"] + fit["slope_log_compute"] * math.log(c))


def _predict_power(fit: dict[str, Any], c: float) -> float:
    return float(fit["intercept"] + fit["coef"] * (c**fit["exponent_k"]))


def _predict_piecewise(fit: dict[str, Any], c: float) -> float:
    x = math.log(c)
    knot = fit.get("knot_log_compute")
    if knot is None:
        return float(fit["intercept"] + fit["slope_left"] * x)
    hinge = max(0.0, x - float(knot))
    # slope_right = slope_left + extra_slope; extra is inferred from left/right.
    extra = float(fit["slope_right"] - fit["slope_left"])
    return float(fit["intercept"] + fit["slope_left"] * x + extra * hinge)


def _loco_scores(c: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    preds: dict[str, list[float]] = {"log_linear": [], "power_law": [], "piecewise_log_linear": []}
    trues: list[float] = []
    for i in range(len(c)):
        mask = np.ones(len(c), dtype=bool)
        mask[i] = False
        c_train, y_train = c[mask], y[mask]
        c_test, y_test = float(c[i]), float(y[i])

        fit_log = _fit_log_linear(c_train, y_train)
        fit_pow = _fit_power_grid(c_train, y_train)
        fit_pwl = _fit_piecewise_log(c_train, y_train)

        p_log = _predict_log_linear(fit_log, c_test)
        p_pow = _predict_power(fit_pow, c_test)
        p_pwl = _predict_piecewise(fit_pwl, c_test)

        rows.append(
            {
                "left_out_compute": c_test,
                "true": y_test,
                "pred_log_linear": p_log,
                "pred_power_law": p_pow,
                "pred_piecewise_log_linear": p_pwl,
            }
        )
        preds["log_linear"].append(p_log)
        preds["power_law"].append(p_pow)
        preds["piecewise_log_linear"].append(p_pwl)
        trues.append(y_test)

    t = np.asarray(trues, dtype=float)
    summary: dict[str, Any] = {}
    for k, pv in preds.items():
        p = np.asarray(pv, dtype=float)
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        summary[k] = {"mae": mae, "rmse": rmse}

    return {"folds": rows, "summary": summary}


def _quantile(vals: Iterable[float], q: float) -> float:
    arr = np.asarray(list(vals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _bootstrap_fits(
    c: np.ndarray,
    y_by_seed: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    # y_by_seed: (n_seeds, n_compute)
    n_seeds = y_by_seed.shape[0]
    if n_seeds == 0:
        return {}

    log_slopes = []
    pow_k = []
    pwl_left = []
    pwl_right = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        y = np.mean(y_by_seed[idx, :], axis=0)
        log_fit = _fit_log_linear(c, y)
        pow_fit = _fit_power_grid(c, y)
        pwl_fit = _fit_piecewise_log(c, y)
        log_slopes.append(float(log_fit["slope_log_compute"]))
        pow_k.append(float(pow_fit["exponent_k"]))
        pwl_left.append(float(pwl_fit["slope_left"]))
        pwl_right.append(float(pwl_fit["slope_right"]))

    return {
        "log_linear": {
            "slope_log_compute_ci95": [_quantile(log_slopes, 0.025), _quantile(log_slopes, 0.975)],
            "slope_log_compute_prob_gt_0": float(np.mean(np.asarray(log_slopes, dtype=float) > 0.0)),
        },
        "power_law": {
            "exponent_k_ci95": [_quantile(pow_k, 0.025), _quantile(pow_k, 0.975)],
            "exponent_k_prob_gt_0": float(np.mean(np.asarray(pow_k, dtype=float) > 0.0)),
        },
        "piecewise_log_linear": {
            "slope_left_ci95": [_quantile(pwl_left, 0.025), _quantile(pwl_left, 0.975)],
            "slope_right_ci95": [_quantile(pwl_right, 0.025), _quantile(pwl_right, 0.975)],
            "slope_right_prob_gt_0": float(np.mean(np.asarray(pwl_right, dtype=float) > 0.0)),
        },
    }


def _parse_group_arg(values: list[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in values:
        if ":" not in item:
            raise SystemExit(f"Invalid --group value '{item}', expected '<compute>:<run_root>'")
        lhs, rhs = item.split(":", 1)
        compute = int(lhs.strip())
        path = Path(rhs.strip()).resolve()
        if not path.exists():
            raise SystemExit(f"Group path does not exist: {path}")
        out[compute] = path
    if len(out) < 3:
        raise SystemExit("Provide at least 3 --group entries for scaling fits.")
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit compute scaling laws from matched run groups")
    p.add_argument(
        "--group",
        action="append",
        default=[],
        help="Compute group as '<compute>:<run_root>' (repeatable)",
    )
    p.add_argument("--baseline", type=int, default=None, help="Baseline compute (default: smallest group)")
    p.add_argument(
        "--metrics",
        default="ensemble_sharpe,ensemble_annret,pair_mean_abs_corr",
        help="Comma-separated metric names",
    )
    p.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap resamples over seeds")
    p.add_argument("--seed", type=int, default=20260215, help="RNG seed")
    p.add_argument("--out", default=None, help="Output JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    group_paths = _parse_group_arg(args.group)
    computes = np.asarray(list(group_paths.keys()), dtype=float)

    baseline = args.baseline if args.baseline is not None else int(min(group_paths.keys()))
    if baseline not in group_paths:
        raise SystemExit(f"Baseline compute {baseline} not found in group list.")

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    default_map = {m.name: m for m in DEFAULT_METRICS}
    metrics: list[MetricSpec] = []
    for name in metric_names:
        if name in default_map:
            metrics.append(default_map[name])
        else:
            raise SystemExit(f"Unsupported metric: {name}")

    groups: dict[int, dict[int, dict[str, float]]] = {}
    for c, root in group_paths.items():
        groups[c] = _collect_group(root)

    common_seeds = sorted(set.intersection(*[set(g.keys()) for g in groups.values()]))
    if not common_seeds:
        raise SystemExit("No common seeds across provided groups.")

    if baseline not in groups:
        raise SystemExit(f"Missing baseline group: {baseline}")

    baseline_idx = list(group_paths.keys()).index(baseline)
    rng = np.random.default_rng(args.seed)

    metric_results: list[dict[str, Any]] = []
    for spec in metrics:
        values = np.full((len(common_seeds), len(group_paths)), np.nan, dtype=float)
        for si, seed in enumerate(common_seeds):
            for ci, c in enumerate(group_paths.keys()):
                values[si, ci] = float(groups[c][seed].get(spec.name, np.nan))

        if spec.higher_is_better:
            oriented = values
        else:
            oriented = -values

        baseline_vals = oriented[:, baseline_idx : baseline_idx + 1]
        improvements = oriented - baseline_vals
        mean_improvement_by_compute = np.mean(improvements, axis=0)

        fit_log = _fit_log_linear(computes, mean_improvement_by_compute)
        fit_pow = _fit_power_grid(computes, mean_improvement_by_compute)
        fit_pwl = _fit_piecewise_log(computes, mean_improvement_by_compute)
        loco = _loco_scores(computes, mean_improvement_by_compute)
        boot = _bootstrap_fits(computes, improvements, n_boot=args.bootstrap, rng=rng)

        metric_results.append(
            {
                "metric": spec.name,
                "higher_is_better": spec.higher_is_better,
                "n_common_seeds": len(common_seeds),
                "compute_levels": [int(x) for x in computes.tolist()],
                "mean_improvement_vs_baseline": {
                    str(int(c)): float(v) for c, v in zip(computes.tolist(), mean_improvement_by_compute.tolist())
                },
                "models": {
                    "log_linear": fit_log,
                    "power_law": fit_pow,
                    "piecewise_log_linear": fit_pwl,
                },
                "predictive_checks_loco": loco,
                "bootstrap_uncertainty": boot,
            }
        )

    out_obj = {
        "schema_version": 1,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "group_roots": {str(k): str(v) for k, v in group_paths.items()},
        "baseline_compute": baseline,
        "common_seeds": common_seeds,
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "metrics": metric_results,
    }

    out_path: Path
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        out_path = Path("artifacts") / "scaling_fits" / f"scaling_law_fit_{stamp}.json"
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

    md_path = out_path.with_suffix(".md")
    lines = [
        f"# Scaling Law Fit: baseline g{baseline}",
        "",
        f"Generated: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"- Common seeds: {len(common_seeds)}",
        f"- Compute levels: {', '.join(str(int(c)) for c in computes.tolist())}",
        "",
    ]
    for m in metric_results:
        lines.append(f"## {m['metric']}")
        lines.append("")
        lines.append("Mean improvement vs baseline:")
        for c in m["compute_levels"]:
            lines.append(f"- g{c}: {m['mean_improvement_vs_baseline'][str(c)]:+.6f}")
        lines.append("")
        log_fit = m["models"]["log_linear"]
        pow_fit = m["models"]["power_law"]
        pwl_fit = m["models"]["piecewise_log_linear"]
        lines.append(
            f"- Log-linear slope (per log-compute): {log_fit['slope_log_compute']:+.6f}"
        )
        lines.append(
            f"- Power-law exponent k: {pow_fit['exponent_k']:+.6f}"
        )
        lines.append(
            f"- Piecewise slopes: left={pwl_fit['slope_left']:+.6f}, right={pwl_fit['slope_right']:+.6f} at knot g{pwl_fit.get('knot_compute', 'na')}"
        )
        cv = m["predictive_checks_loco"]["summary"]
        lines.append(
            f"- LOCO MAE: log={cv['log_linear']['mae']:.6f}, power={cv['power_law']['mae']:.6f}, piecewise={cv['piecewise_log_linear']['mae']:.6f}"
        )
        bu = m["bootstrap_uncertainty"]
        lines.append(
            f"- Bootstrap CI95 log-slope: [{bu['log_linear']['slope_log_compute_ci95'][0]:+.6f}, {bu['log_linear']['slope_log_compute_ci95'][1]:+.6f}] (P>0={bu['log_linear']['slope_log_compute_prob_gt_0']:.3f})"
        )
        lines.append(
            f"- Bootstrap CI95 power-k: [{bu['power_law']['exponent_k_ci95'][0]:+.6f}, {bu['power_law']['exponent_k_ci95'][1]:+.6f}] (P>0={bu['power_law']['exponent_k_prob_gt_0']:.3f})"
        )
        lines.append(
            f"- Bootstrap CI95 piecewise right-slope: [{bu['piecewise_log_linear']['slope_right_ci95'][0]:+.6f}, {bu['piecewise_log_linear']['slope_right_ci95'][1]:+.6f}] (P>0={bu['piecewise_log_linear']['slope_right_prob_gt_0']:.3f})"
        )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[scaling-fit] Wrote JSON -> {out_path}")
    print(f"[scaling-fit] Wrote Markdown -> {md_path}")


if __name__ == "__main__":
    main()
