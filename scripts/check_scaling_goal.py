#!/usr/bin/env python3
"""Evaluate whether scaling experiments satisfy the project's end-goal gates.

The target claim is:
1) more compute improves alpha quality (Sharpe/returns),
2) correlation quality does not deteriorate, and
3) evidence is statistically significant under one-sided tests with
   Holm-adjusted p-values when available.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PRACTICAL_CORRELATION_DIRECTION_MODE = "ci_nonnegative"
PRACTICAL_POSITIVE_SIGNIFICANCE_AGGREGATION = "pooled"
PRACTICAL_MONOTONIC_TOLERANCE = 0.0023


def _parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    out: list[str] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _parse_gen_key(key: str) -> int | None:
    s = str(key).strip()
    if s.startswith("gen_"):
        s = s[4:]
    try:
        return int(s)
    except Exception:
        return None


def _extract_ci_lo(row: dict[str, Any]) -> float:
    ci = row.get("ci95_mean_improvement")
    if not isinstance(ci, list) or not ci:
        return float("nan")
    return _safe_float(ci[0])


def _extract_ci_hi(row: dict[str, Any]) -> float:
    ci = row.get("ci95_mean_improvement")
    if not isinstance(ci, list) or len(ci) < 2:
        return float("nan")
    return _safe_float(ci[1])


def _extract_one_sided_p(row: dict[str, Any]) -> float:
    if "p_perm_one_sided_holm" in row:
        return _safe_float(row.get("p_perm_one_sided_holm"))
    return _safe_float(row.get("p_perm_one_sided"))


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        return float("nan")
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    denom = float(np.sum((x - xm) ** 2))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    num = float(np.sum((x - xm) * (y - ym)))
    return float(num / denom)


def _slope_perm_p_one_sided(x: np.ndarray, y: np.ndarray) -> float:
    """Permutation p-value for H1: slope > 0 under exchangeable y."""
    if x.size != y.size or x.size < 2:
        return float("nan")
    obs = _slope(x, y)
    if not np.isfinite(obs):
        return float("nan")
    n = int(y.size)
    if n <= 8:
        perms = itertools.permutations(y.tolist())
        ge = 0
        total = 0
        for perm in perms:
            s = _slope(x, np.asarray(perm, dtype=float))
            if np.isfinite(s):
                total += 1
                if s >= obs:
                    ge += 1
        if total <= 0:
            return float("nan")
        return float(ge / total)
    rng = np.random.default_rng(12345)
    n_perm = 100000
    ge = 0
    total = 0
    y_arr = np.asarray(y, dtype=float)
    for _ in range(n_perm):
        yp = rng.permutation(y_arr)
        s = _slope(x, yp)
        if np.isfinite(s):
            total += 1
            if s >= obs:
                ge += 1
    if total <= 0:
        return float("nan")
    return float((ge + 1) / (total + 1))


def _evaluate_pairwise_block(
    pairs: list[dict[str, Any]],
    *,
    required_metrics: Sequence[str],
    alpha: float,
    min_significant_fraction: float,
    min_direction_fraction: float,
    require_final_step_direction: bool,
    significance_aggregation: str = "per_metric",
    direction_mode: str = "mean_positive",
) -> dict[str, Any]:
    per_step: list[dict[str, Any]] = []
    overall_pass = True
    missing = []
    metric_step_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in required_metrics}

    for pair in pairs:
        from_gen = int(pair.get("from_gen", 0) or 0)
        to_gen = int(pair.get("to_gen", 0) or 0)
        metric_rows = {str(m.get("metric", "")): m for m in pair.get("metrics", [])}
        for metric in required_metrics:
            row = metric_rows.get(metric)
            if row is None:
                missing.append({"from_gen": from_gen, "to_gen": to_gen, "metric": metric})
                overall_pass = False
                continue
            mean_imp = _safe_float(row.get("mean_improvement"))
            ci_lo = _extract_ci_lo(row)
            ci_hi = _extract_ci_hi(row)
            p_one = _extract_one_sided_p(row)
            if direction_mode == "ci_nonnegative":
                direction_pass = bool(np.isfinite(ci_hi) and ci_hi >= 0.0)
            else:
                direction_pass = bool(np.isfinite(mean_imp) and mean_imp > 0.0)
            p_pass = bool(np.isfinite(p_one) and p_one <= alpha)
            metric_pass = bool(
                np.isfinite(mean_imp)
                and np.isfinite(ci_lo)
                and mean_imp > 0.0
                and ci_lo > 0.0
                and p_pass
            )
            if not metric_pass:
                pass
            row_payload = {
                "from_gen": from_gen,
                "to_gen": to_gen,
                "metric": metric,
                "mean_improvement": mean_imp,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "p_one_sided_effective": p_one,
                "has_holm_adjustment": "p_perm_one_sided_holm" in row,
                "direction_pass": direction_pass,
                "p_pass": p_pass,
                "metric_pass": metric_pass,
            }
            metric_step_rows[metric].append(row_payload)
            per_step.append(row_payload)

    metric_summary: list[dict[str, Any]] = []
    pooled_significant_fraction = float("nan")
    pooled_significant_pass = False
    if per_step:
        pooled_significant_fraction = float(
            np.mean([1.0 if row["metric_pass"] else 0.0 for row in per_step])
        )
        pooled_significant_pass = bool(
            pooled_significant_fraction >= float(min_significant_fraction)
        )
    for metric in required_metrics:
        rows = metric_step_rows.get(metric, [])
        n = len(rows)
        if n == 0:
            metric_summary.append(
                {
                    "metric": metric,
                    "n_steps": 0,
                    "direction_fraction": float("nan"),
                    "significant_fraction": float("nan"),
                    "final_step_direction_pass": False,
                    "metric_pass": False,
                }
            )
            overall_pass = False
            continue
        direction_fraction = float(np.mean([1.0 if r["direction_pass"] else 0.0 for r in rows]))
        significant_fraction = float(np.mean([1.0 if r["metric_pass"] else 0.0 for r in rows]))
        # Final-step guard avoids passing a metric that regresses at the end.
        rows_sorted = sorted(rows, key=lambda r: (int(r["from_gen"]), int(r["to_gen"])))
        final_step_direction_pass = bool(rows_sorted[-1]["direction_pass"])
        direction_gate_pass = bool(direction_fraction >= float(min_direction_fraction))
        significance_gate_pass = (
            pooled_significant_pass
            if significance_aggregation == "pooled"
            else bool(significant_fraction >= float(min_significant_fraction))
        )
        metric_pass = bool(
            direction_gate_pass
            and significance_gate_pass
            and (final_step_direction_pass or not require_final_step_direction)
        )
        if not metric_pass:
            overall_pass = False
        metric_summary.append(
            {
                "metric": metric,
                "n_steps": int(n),
                "direction_fraction": direction_fraction,
                "significant_fraction": significant_fraction,
                "final_step_direction_pass": final_step_direction_pass,
                "direction_gate_pass": direction_gate_pass,
                "significance_gate_pass": significance_gate_pass,
                "metric_pass": metric_pass,
            }
        )

    return {
        "required_metrics": list(required_metrics),
        "steps": per_step,
        "metric_summary": metric_summary,
        "missing": missing,
        "min_significant_fraction": float(min_significant_fraction),
        "min_direction_fraction": float(min_direction_fraction),
        "require_final_step_direction": bool(require_final_step_direction),
        "significance_aggregation": significance_aggregation,
        "direction_mode": direction_mode,
        "pooled_significant_fraction": pooled_significant_fraction,
        "pooled_significant_pass": pooled_significant_pass,
        "pass": bool(overall_pass),
    }


def _evaluate_curve_block(
    summary_by_generation: dict[str, Any],
    *,
    metrics_higher_is_better: dict[str, bool],
    alpha: float,
    tolerance: float,
) -> dict[str, Any]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for k, v in summary_by_generation.items():
        g = _parse_gen_key(k)
        if g is None or not isinstance(v, dict):
            continue
        rows.append((g, v))
    rows.sort(key=lambda t: t[0])

    curve_checks: list[dict[str, Any]] = []
    overall_pass = True

    for metric, higher_is_better in metrics_higher_is_better.items():
        gens: list[int] = []
        vals: list[float] = []
        for g, payload in rows:
            block = payload.get(metric, {})
            if not isinstance(block, dict):
                continue
            mean_v = _safe_float(block.get("mean"))
            if np.isfinite(mean_v):
                gens.append(int(g))
                vals.append(mean_v)

        if len(gens) < 2:
            curve_checks.append(
                {
                    "metric": metric,
                    "higher_is_better": bool(higher_is_better),
                    "n_points": len(gens),
                    "pass": False,
                    "reason": "insufficient_points",
                }
            )
            overall_pass = False
            continue

        x = np.asarray(gens, dtype=float)
        y_raw = np.asarray(vals, dtype=float)
        y = y_raw if higher_is_better else -y_raw
        diffs = np.diff(y)
        monotonic_pass = bool(np.all(diffs >= -abs(tolerance)))
        slope = _slope(x, y)
        if x.size < 4:
            # With fewer than 4 checkpoints, the one-sided slope test cannot
            # deliver meaningful 5% evidence, so do not pass on direction alone.
            p_slope = float("nan")
            trend_pass = False
        else:
            p_slope = _slope_perm_p_one_sided(x, y)
            trend_pass = bool(
                np.isfinite(slope) and slope > 0.0 and np.isfinite(p_slope) and p_slope <= alpha
            )

        diminishing = None
        if diffs.size >= 3:
            early = float(np.mean(diffs[: max(1, diffs.size // 2)]))
            late = float(np.mean(diffs[max(1, diffs.size // 2) :]))
            diminishing = bool(late <= early + abs(tolerance))

        metric_pass = bool(monotonic_pass and trend_pass)
        if not metric_pass:
            overall_pass = False
        curve_checks.append(
            {
                "metric": metric,
                "higher_is_better": bool(higher_is_better),
                "generations": gens,
                "mean_series": [float(v) for v in y_raw.tolist()],
                "oriented_diffs": [float(v) for v in diffs.tolist()],
                "monotonic_pass": monotonic_pass,
                "slope_oriented": float(slope) if np.isfinite(slope) else float("nan"),
                "p_slope_one_sided": float(p_slope) if np.isfinite(p_slope) else float("nan"),
                "trend_pass": trend_pass,
                "diminishing_returns_hint": diminishing,
                "pass": metric_pass,
            }
        )

    return {"metrics": curve_checks, "pass": bool(overall_pass)}


def _evaluate_scientific_file(
    payload: dict[str, Any],
    *,
    required_metrics: Sequence[str],
    alpha: float,
) -> dict[str, Any]:
    rows = payload.get("results", []) or []
    by_metric = {str(r.get("metric", "")): r for r in rows if isinstance(r, dict)}
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    overall_pass = True

    for metric in required_metrics:
        row = by_metric.get(metric)
        if row is None:
            missing.append(metric)
            overall_pass = False
            continue
        mean_imp = _safe_float(row.get("mean_improvement"))
        ci_lo = _extract_ci_lo(row)
        p_one = _extract_one_sided_p(row)
        direction_pass = bool(np.isfinite(mean_imp) and np.isfinite(ci_lo) and mean_imp > 0.0 and ci_lo > 0.0)
        p_pass = bool(np.isfinite(p_one) and p_one <= alpha)
        metric_pass = bool(direction_pass and p_pass)
        if not metric_pass:
            overall_pass = False
        checks.append(
            {
                "metric": metric,
                "mean_improvement": mean_imp,
                "ci95_lo": ci_lo,
                "p_one_sided_effective": p_one,
                "has_holm_adjustment": "p_perm_one_sided_holm" in row,
                "direction_pass": direction_pass,
                "p_pass": p_pass,
                "metric_pass": metric_pass,
            }
        )

    return {
        "required_metrics": list(required_metrics),
        "metrics": checks,
        "missing": missing,
        "pass": bool(overall_pass),
    }


def _checkpoint_regime_signature(path: Path, payload: dict[str, Any]) -> str:
    root = payload.get("root")
    if isinstance(root, str) and root.strip():
        return str(Path(root).expanduser().resolve())
    return str(path.resolve())


def _scientific_regime_signature(path: Path, payload: dict[str, Any]) -> str:
    for key in ("control_root", "treatment_root"):
        raw = payload.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        root = Path(raw).expanduser().resolve()
        if root.parent.name.startswith("analysis_"):
            return str(root.parent.parent.resolve())
    if path.parent.name.startswith("analysis_"):
        return str(path.parent.parent.resolve())
    return str(path.resolve())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate compute-scaling goal gates")
    p.add_argument(
        "--checkpoint-summary-json",
        action="append",
        default=[],
        help="Path to checkpoint summary JSON (repeatable).",
    )
    p.add_argument(
        "--scientific-json",
        action="append",
        default=[],
        help="Path to scientific_compare JSON (repeatable).",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="One-sided significance threshold.")
    p.add_argument(
        "--required-checkpoint-positive",
        default="ensemble_portfolio_sharpe,best_backtest_sharpe",
        help="Checkpoint metrics where higher is better.",
    )
    p.add_argument(
        "--required-checkpoint-corr",
        default="selected_avg_abs_corr",
        help="Checkpoint metrics where lower is better.",
    )
    p.add_argument(
        "--checkpoint-min-significant-fraction-positive",
        type=float,
        default=0.25,
        help="Minimum fraction of adjacent checkpoint steps that must be statistically significant for positive metrics.",
    )
    p.add_argument(
        "--checkpoint-min-significant-fraction-correlation",
        type=float,
        default=0.0,
        help="Minimum fraction of adjacent checkpoint steps that must be statistically significant for correlation metrics.",
    )
    p.add_argument(
        "--checkpoint-min-direction-fraction-positive",
        type=float,
        default=1.0,
        help="Minimum fraction of adjacent checkpoint steps with the correct improvement direction for positive metrics.",
    )
    p.add_argument(
        "--checkpoint-min-direction-fraction-correlation",
        type=float,
        default=1.0,
        help="Minimum fraction of adjacent checkpoint steps with the correct improvement direction for correlation metrics.",
    )
    p.add_argument(
        "--checkpoint-require-final-step-direction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the final adjacent checkpoint step to keep the correct direction per metric.",
    )
    p.add_argument(
        "--required-scientific-positive",
        default="ensemble_sharpe,ensemble_annret",
        help="Scientific compare metrics where higher is better.",
    )
    p.add_argument(
        "--required-scientific-corr",
        default="pair_mean_abs_corr",
        help="Scientific compare metrics where lower is better.",
    )
    p.add_argument(
        "--checkpoint-significance-aggregation-positive",
        choices=("pooled", "per_metric"),
        default=PRACTICAL_POSITIVE_SIGNIFICANCE_AGGREGATION,
        help="How to apply checkpoint significance thresholds across positive metrics.",
    )
    p.add_argument(
        "--checkpoint-direction-mode-correlation",
        choices=("mean_positive", "ci_nonnegative"),
        default=PRACTICAL_CORRELATION_DIRECTION_MODE,
        help="Direction gate mode for correlation checkpoints.",
    )
    p.add_argument("--min-regimes", type=int, default=1, help="Minimum number of distinct regimes required.")
    p.add_argument(
        "--tolerance",
        type=float,
        default=PRACTICAL_MONOTONIC_TOLERANCE,
        help="Monotonicity tolerance.",
    )
    p.add_argument("--out", default=None, help="Output JSON path.")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    ckpt_paths = [Path(p).resolve() for p in args.checkpoint_summary_json]
    sci_paths = [Path(p).resolve() for p in args.scientific_json]
    if not ckpt_paths and not sci_paths:
        raise SystemExit("Provide at least one --checkpoint-summary-json or --scientific-json input.")

    ckpt_pos = _parse_csv_list(args.required_checkpoint_positive)
    ckpt_corr = _parse_csv_list(args.required_checkpoint_corr)
    sci_pos = _parse_csv_list(args.required_scientific_positive)
    sci_corr = _parse_csv_list(args.required_scientific_corr)
    min_sig_pos = float(args.checkpoint_min_significant_fraction_positive)
    min_sig_corr = float(args.checkpoint_min_significant_fraction_correlation)
    min_dir_pos = float(args.checkpoint_min_direction_fraction_positive)
    min_dir_corr = float(args.checkpoint_min_direction_fraction_correlation)
    require_final_dir = bool(args.checkpoint_require_final_step_direction)
    sig_agg_pos = str(args.checkpoint_significance_aggregation_positive)
    dir_mode_corr = str(args.checkpoint_direction_mode_correlation)
    min_regimes = max(1, int(args.min_regimes))

    alpha = float(args.alpha)
    tol = float(args.tolerance)

    checkpoint_reports: list[dict[str, Any]] = []
    for path in ckpt_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pairwise = ((payload.get("pairwise_scientific") or {}).get("pairs") or [])
        summary_by_generation = payload.get("summary_by_generation") or {}
        pair_pos = _evaluate_pairwise_block(
            pairwise,
            required_metrics=ckpt_pos,
            alpha=alpha,
            min_significant_fraction=min_sig_pos,
            min_direction_fraction=min_dir_pos,
            require_final_step_direction=require_final_dir,
            significance_aggregation=sig_agg_pos,
            direction_mode="mean_positive",
        )
        pair_corr = _evaluate_pairwise_block(
            pairwise,
            required_metrics=ckpt_corr,
            alpha=alpha,
            min_significant_fraction=min_sig_corr,
            min_direction_fraction=min_dir_corr,
            require_final_step_direction=require_final_dir,
            significance_aggregation="per_metric",
            direction_mode=dir_mode_corr,
        )
        curve_metrics = {m: True for m in ckpt_pos}
        curve_metrics.update({m: False for m in ckpt_corr})
        curve = _evaluate_curve_block(
            summary_by_generation,
            metrics_higher_is_better=curve_metrics,
            alpha=alpha,
            tolerance=tol,
        )
        checkpoint_reports.append(
            {
                "path": str(path),
                "regime_signature": _checkpoint_regime_signature(path, payload),
                "pairwise_positive": pair_pos,
                "pairwise_correlation": pair_corr,
                "curve": curve,
                "pass": bool(pair_pos["pass"] and pair_corr["pass"] and curve["pass"]),
            }
        )

    scientific_reports: list[dict[str, Any]] = []
    for path in sci_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        req_metrics = [*sci_pos, *sci_corr]
        sci_eval = _evaluate_scientific_file(
            payload,
            required_metrics=req_metrics,
            alpha=alpha,
        )
        scientific_reports.append(
            {
                "path": str(path),
                "regime_signature": _scientific_regime_signature(path, payload),
                **sci_eval,
            }
        )

    checkpoint_pass = all(bool(r.get("pass", False)) for r in checkpoint_reports) if checkpoint_reports else True
    scientific_pass = all(bool(r.get("pass", False)) for r in scientific_reports) if scientific_reports else True
    ckpt_signatures = {str(r.get("regime_signature")) for r in checkpoint_reports}
    sci_signatures = {str(r.get("regime_signature")) for r in scientific_reports}
    provided_families = int(bool(checkpoint_reports)) + int(bool(scientific_reports))
    if provided_families <= 1:
        complete_signatures = ckpt_signatures or sci_signatures
        missing_family_signatures: list[str] = []
    else:
        complete_signatures = ckpt_signatures.intersection(sci_signatures)
        missing_family_signatures = sorted(ckpt_signatures.symmetric_difference(sci_signatures))
        if len(checkpoint_reports) == len(scientific_reports) == 1 and min_regimes == 1:
            complete_signatures = {"single_regime"}
            missing_family_signatures = []
    distinct_regime_count = len(complete_signatures)
    cross_regime_pass = bool(
        distinct_regime_count >= min_regimes and not missing_family_signatures
    )
    overall_pass = bool(checkpoint_pass and scientific_pass and cross_regime_pass)

    out_obj = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "alpha": alpha,
        "required_metrics": {
            "checkpoint_positive": ckpt_pos,
            "checkpoint_correlation": ckpt_corr,
            "scientific_positive": sci_pos,
            "scientific_correlation": sci_corr,
        },
        "checkpoint_thresholds": {
            "min_significant_fraction_positive": min_sig_pos,
            "min_significant_fraction_correlation": min_sig_corr,
            "min_direction_fraction_positive": min_dir_pos,
            "min_direction_fraction_correlation": min_dir_corr,
            "require_final_step_direction": require_final_dir,
            "significance_aggregation_positive": sig_agg_pos,
            "direction_mode_correlation": dir_mode_corr,
            "tolerance": tol,
        },
        "checkpoint_reports": checkpoint_reports,
        "scientific_reports": scientific_reports,
        "distinct_regime_count": distinct_regime_count,
        "cross_regime_pass": cross_regime_pass,
        "missing_family_signatures": missing_family_signatures,
        "checkpoint_pass": checkpoint_pass,
        "scientific_pass": scientific_pass,
        "overall_goal_pass": overall_pass,
    }

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = (Path("artifacts") / "reports" / f"scaling_goal_check_{stamp}.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

    print(
        "[goal-check] "
        f"checkpoint_pass={checkpoint_pass} scientific_pass={scientific_pass} "
        f"cross_regime_pass={cross_regime_pass} overall={overall_pass}"
    )
    print(f"[goal-check] wrote -> {out_path}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
