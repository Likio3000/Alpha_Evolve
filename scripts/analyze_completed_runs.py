#!/usr/bin/env python3
"""Analyze completed run_* folders for scaling/correlation evidence.

Useful when long parallel shard jobs are still in progress: this script reads
only runs that already have SUMMARY.json and computes:
1) per-run final metrics (for paired control/treatment comparisons), and
2) checkpoint 45/60/90 style paired improvements from checkpoint backtests.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SEED_RE = re.compile(r"seed(\d+)")


def _parse_seed(name: str) -> int | None:
    m = SEED_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _pairwise_corr_stats(corr: pd.DataFrame, members: Sequence[str]) -> dict[str, Any]:
    members = [m for m in members if m in corr.index]
    if len(members) < 2:
        return {"k": len(members), "avg_abs_corr": 0.0, "max_abs_corr": 0.0}
    mat = corr.loc[members, members].to_numpy(dtype=float)
    mat = np.abs(mat)
    vals = mat[np.triu_indices(mat.shape[0], k=1)]
    vals = vals[pd.notna(vals)]
    if vals.size == 0:
        return {"k": len(members), "avg_abs_corr": 0.0, "max_abs_corr": 0.0}
    return {
        "k": int(len(members)),
        "avg_abs_corr": float(vals.mean()),
        "max_abs_corr": float(vals.max()),
    }


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
                sub = cm.loc[members, members]
                corr_stats = _pairwise_corr_stats(sub, members)
                ensemble_mean_abs_corr = float(corr_stats["avg_abs_corr"])
                ensemble_max_abs_corr = float(corr_stats["max_abs_corr"])

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


def _collect_backtest_dir_summary(bt_dir: Path) -> dict[str, Any]:
    best_sharpe = None
    best_alpha = None
    raw_top_members: list[str] = []
    summary_csv = None
    for cand in bt_dir.glob("backtest_summary_top*.csv"):
        summary_csv = cand
        break
    if summary_csv and summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            df = df.sort_values("Sharpe", ascending=False)
            if len(df) > 0 and "Sharpe" in df.columns:
                best_sharpe = float(df.iloc[0]["Sharpe"])
                best_alpha = str(df.iloc[0].get("AlphaID", ""))
            raw_top_members = [str(x) for x in df.head(min(5, int(len(df)))).get("AlphaID", []).tolist() if x]
        except Exception:
            pass

    ens_members: list[str] = []
    ens_path = bt_dir / "ensemble_selection.json"
    if ens_path.exists():
        try:
            ens_members = list((json.loads(ens_path.read_text(encoding="utf-8"))).get("members") or [])
        except Exception:
            ens_members = []

    corr_df: pd.DataFrame | None = None
    corr_path = bt_dir / "return_corr_matrix.csv"
    if corr_path.exists():
        try:
            corr_df = pd.read_csv(corr_path, index_col=0)
        except Exception:
            corr_df = None

    corr_stats_ens = _pairwise_corr_stats(corr_df, ens_members) if corr_df is not None else None
    corr_stats_raw = _pairwise_corr_stats(corr_df, raw_top_members) if corr_df is not None else None

    ens_port_sharpe = None
    ens_csv = bt_dir / "backtest_summary_ensemble.csv"
    if ens_csv.exists():
        try:
            df = pd.read_csv(ens_csv)
            if len(df) > 0 and "Sharpe" in df.columns:
                ens_port_sharpe = float(df.iloc[0]["Sharpe"])
        except Exception:
            ens_port_sharpe = None

    return {
        "best_backtest_sharpe": best_sharpe,
        "best_alpha": best_alpha,
        "ensemble_members": ens_members,
        "ensemble_portfolio_sharpe": ens_port_sharpe,
        "raw_top_members": raw_top_members,
        "corr_selected": corr_stats_ens,
        "corr_raw_topk": corr_stats_raw,
    }


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 20000,
    seed: int = 123,
) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        v = float(vals[0])
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = np.mean(vals[idx], axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _paired_sign_flip_pvalues(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return 1.0, 1.0
    obs = float(np.mean(vals))
    if n <= 20:
        all_means: list[float] = []
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


def _checkpoint_pairwise_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"schema_version": 1, "pairs": []}
    df = pd.DataFrame(list(rows))
    if df.empty:
        return {"schema_version": 1, "pairs": []}
    generations = sorted(int(x) for x in df["generation"].dropna().unique().tolist())
    metrics: list[tuple[str, bool]] = [
        ("best_backtest_sharpe", True),
        ("ensemble_portfolio_sharpe", True),
        ("selected_avg_abs_corr", False),
    ]
    pairs: list[dict[str, Any]] = []
    for g0, g1 in zip(generations, generations[1:]):
        pair_payload: dict[str, Any] = {"from_gen": int(g0), "to_gen": int(g1), "metrics": []}
        left = df[df["generation"] == g0].set_index("seed")
        right = df[df["generation"] == g1].set_index("seed")
        common = sorted(set(left.index).intersection(set(right.index)))
        for metric, higher_is_better in metrics:
            deltas: list[float] = []
            for s in common:
                try:
                    a = float(left.loc[s, metric])
                    b = float(right.loc[s, metric])
                except Exception:
                    continue
                if not np.isfinite(a) or not np.isfinite(b):
                    continue
                deltas.append((b - a) if higher_is_better else (a - b))
            if not deltas:
                continue
            arr = np.asarray(deltas, dtype=float)
            eps = 1e-12
            pos = int(np.sum(arr > eps))
            neg = int(np.sum(arr < -eps))
            zero = int(arr.size - pos - neg)
            non_neg = float(np.mean(arr >= -eps)) if arr.size else float("nan")
            ci_lo, ci_hi = _bootstrap_mean_ci(arr, alpha=0.05, n_boot=20000, seed=123)
            p_one, p_two = _paired_sign_flip_pvalues(arr)
            pair_payload["metrics"].append(
                {
                    "metric": metric,
                    "higher_is_better": bool(higher_is_better),
                    "n": int(arr.size),
                    "mean_improvement": float(np.mean(arr)),
                    "median_improvement": float(np.median(arr)),
                    "ci95_mean_improvement": [float(ci_lo), float(ci_hi)],
                    "positives": pos,
                    "zeros": zero,
                    "negatives": neg,
                    "non_negative_fraction": non_neg,
                    "monotonic_non_decreasing_all": bool(neg == 0),
                    "scientific_pass_non_decrease": bool(neg == 0 and pos > 0),
                    "p_perm_one_sided": float(p_one),
                    "p_perm_two_sided": float(p_two),
                    "scientific_pass": bool(np.isfinite(ci_lo) and ci_lo > 0 and p_one <= 0.05),
                }
            )
        pairs.append(pair_payload)
    return {"schema_version": 1, "pairs": pairs}


def _aggregate(values: Iterable[float]) -> dict[str, Any]:
    xs = pd.Series([v for v in values if v is not None and pd.notna(v)], dtype="float64")
    if xs.empty:
        return {"count": 0}
    return {
        "count": int(xs.size),
        "mean": float(xs.mean()),
        "median": float(xs.median()),
        "min": float(xs.min()),
        "max": float(xs.max()),
        "p25": float(xs.quantile(0.25)),
        "p75": float(xs.quantile(0.75)),
    }


def _parse_int_list(spec: str | None) -> list[int]:
    if spec is None:
        return []
    s = str(spec).strip()
    if not s:
        return []
    out: list[int] = []
    for token in s.split(","):
        tok = token.strip()
        if not tok:
            continue
        out.append(int(tok))
    return sorted(set(x for x in out if x > 0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze completed run_* folders")
    p.add_argument(
        "--runs-root",
        action="append",
        default=[],
        help="Root directory to scan recursively for run_* folders (repeatable)",
    )
    p.add_argument(
        "--checkpoint-gens",
        default="45,60,90",
        help="Checkpoint generations to evaluate from run_dir/checkpoint_backtests/gen_XXX",
    )
    p.add_argument("--outdir", default="artifacts/run_analysis")
    p.add_argument("--name", default=None, help="Output basename (without extension)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    roots = [Path(p).expanduser().resolve() for p in args.runs_root]
    if not roots:
        raise SystemExit("Provide at least one --runs-root path")
    for r in roots:
        if not r.exists():
            raise SystemExit(f"runs root does not exist: {r}")

    checkpoint_gens = _parse_int_list(args.checkpoint_gens)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = args.name or f"completed_runs_{stamp}"

    run_dirs: list[Path] = []
    for root in roots:
        for p in sorted(root.rglob("run_*")):
            if not p.is_dir():
                continue
            if (p / "SUMMARY.json").exists():
                run_dirs.append(p)

    # Stable dedupe by resolved path.
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in run_dirs:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        dedup.append(p)
    run_dirs = dedup
    if not run_dirs:
        raise SystemExit("No completed run_* directories found (missing SUMMARY.json).")

    final_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for run_dir in sorted(run_dirs):
        seed = _parse_seed(run_dir.name)
        if seed is None:
            continue
        try:
            final_metrics = _collect_run_metrics(run_dir)
        except Exception:
            continue
        row = {"seed": int(seed), "run_dir": str(run_dir.resolve())}
        row.update(final_metrics)
        final_rows.append(row)

        ckpt_root = run_dir / "checkpoint_backtests"
        if not ckpt_root.exists():
            continue
        gens = checkpoint_gens if checkpoint_gens else sorted(
            int(p.name.replace("gen_", ""))
            for p in ckpt_root.glob("gen_*")
            if p.is_dir() and p.name.replace("gen_", "").isdigit()
        )
        for g in gens:
            bt_dir = ckpt_root / f"gen_{int(g):03d}"
            if not bt_dir.exists():
                continue
            rec = {"seed": int(seed), "generation": int(g), "run_dir": str(run_dir.resolve())}
            rec2 = _collect_backtest_dir_summary(bt_dir)
            rec["best_alpha"] = rec2.get("best_alpha")
            rec["best_backtest_sharpe"] = rec2.get("best_backtest_sharpe")
            rec["ensemble_portfolio_sharpe"] = rec2.get("ensemble_portfolio_sharpe")
            corr_sel = rec2.get("corr_selected") or {}
            corr_raw = rec2.get("corr_raw_topk") or {}
            rec["ensemble_k"] = corr_sel.get("k")
            rec["selected_avg_abs_corr"] = corr_sel.get("avg_abs_corr")
            rec["selected_max_abs_corr"] = corr_sel.get("max_abs_corr")
            rec["raw_topk_k"] = corr_raw.get("k")
            rec["raw_topk_avg_abs_corr"] = corr_raw.get("avg_abs_corr")
            rec["raw_topk_max_abs_corr"] = corr_raw.get("max_abs_corr")
            checkpoint_rows.append(rec)

    final_df = pd.DataFrame(final_rows).sort_values(["seed", "run_dir"])
    final_csv = outdir / f"{name}_final_metrics.csv"
    final_df.to_csv(final_csv, index=False)

    final_summary = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "roots": [str(r) for r in roots],
        "n_runs": int(len(final_df)),
        "seeds": sorted(int(x) for x in final_df["seed"].dropna().unique().tolist()) if "seed" in final_df else [],
        "best_sharpe": _aggregate(final_df["best_sharpe"].tolist()) if "best_sharpe" in final_df else {"count": 0},
        "ensemble_sharpe": _aggregate(final_df["ensemble_sharpe"].tolist()) if "ensemble_sharpe" in final_df else {"count": 0},
        "pair_mean_abs_corr": _aggregate(final_df["pair_mean_abs_corr"].tolist()) if "pair_mean_abs_corr" in final_df else {"count": 0},
        "ensemble_mean_abs_corr": _aggregate(final_df["ensemble_mean_abs_corr"].tolist()) if "ensemble_mean_abs_corr" in final_df else {"count": 0},
    }
    final_summary_json = outdir / f"{name}_final_summary.json"
    final_summary_json.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    if checkpoint_rows:
        cp_df = pd.DataFrame(checkpoint_rows).sort_values(["seed", "generation"])
        cp_csv = outdir / f"{name}_checkpoint_metrics.csv"
        cp_df.to_csv(cp_csv, index=False)
        by_gen: dict[str, Any] = {}
        for g in sorted(int(x) for x in cp_df["generation"].dropna().unique().tolist()):
            gdf = cp_df[cp_df["generation"] == g]
            by_gen[f"gen_{g:03d}"] = {
                "n": int(len(gdf)),
                "best_backtest_sharpe": _aggregate(gdf["best_backtest_sharpe"].tolist()),
                "ensemble_portfolio_sharpe": _aggregate(gdf["ensemble_portfolio_sharpe"].tolist()),
                "selected_avg_abs_corr": _aggregate(gdf["selected_avg_abs_corr"].tolist()),
            }
        cp_summary = {
            "schema_version": 1,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_gens": checkpoint_gens,
            "n_rows": int(len(cp_df)),
            "summary_by_generation": by_gen,
            "pairwise_scientific": _checkpoint_pairwise_report(checkpoint_rows),
            "checkpoint_metrics_csv": str(cp_csv),
        }
        cp_summary_json = outdir / f"{name}_checkpoint_summary.json"
        cp_summary_json.write_text(json.dumps(cp_summary, indent=2), encoding="utf-8")
        print(f"[analyze] Wrote checkpoint summary -> {cp_summary_json}")

    print(f"[analyze] Wrote final metrics -> {final_csv}")
    print(f"[analyze] Wrote final summary -> {final_summary_json}")
    print(f"[analyze] Completed runs analyzed: {len(final_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
