#!/usr/bin/env python3
"""Aggregate sharded benchmark_sp500 outputs into one scientific payload.

Designed for directories produced by scripts/benchmark_sp500_parallel.sh.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
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


def _checkpoint_pairwise_report(cp_df: pd.DataFrame) -> dict[str, Any]:
    if cp_df.empty or "generation" not in cp_df.columns or "seed" not in cp_df.columns:
        return {"schema_version": 1, "pairs": []}
    generations = sorted(int(x) for x in cp_df["generation"].dropna().unique().tolist())
    metrics: list[tuple[str, bool]] = [
        ("best_backtest_sharpe", True),
        ("ensemble_portfolio_sharpe", True),
        ("selected_avg_abs_corr", False),
    ]
    pairs: list[dict[str, Any]] = []
    for g0, g1 in zip(generations, generations[1:]):
        pair_payload: dict[str, Any] = {"from_gen": int(g0), "to_gen": int(g1), "metrics": []}
        left = cp_df[cp_df["generation"] == g0].set_index("seed")
        right = cp_df[cp_df["generation"] == g1].set_index("seed")
        common = sorted(set(left.index).intersection(set(right.index)))
        for metric, higher_is_better in metrics:
            if metric not in left.columns or metric not in right.columns:
                continue
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
                    "p_perm_one_sided": float(p_one),
                    "p_perm_two_sided": float(p_two),
                    "scientific_pass": bool(np.isfinite(ci_lo) and ci_lo > 0 and p_one <= 0.05),
                }
            )
        pairs.append(pair_payload)
    return {"schema_version": 1, "pairs": pairs}


def _find_benchmark_dirs(root: Path) -> list[Path]:
    bench_dirs = sorted([p for p in root.glob("sp500_bench_*") if p.is_dir()])
    for shard in sorted([p for p in root.glob("shard_*") if p.is_dir()]):
        bench_dirs.extend(sorted([p for p in shard.glob("sp500_bench_*") if p.is_dir()]))
    # Stable dedupe while preserving order.
    out: list[Path] = []
    seen: set[str] = set()
    for p in bench_dirs:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate sharded benchmark outputs")
    p.add_argument("--root", required=True, help="Root from benchmark_sp500_parallel.sh")
    p.add_argument(
        "--outdir",
        default=None,
        help="Where to write aggregate files (default: <root>/aggregate)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"--root does not exist: {root}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (root / "aggregate")
    outdir.mkdir(parents=True, exist_ok=True)

    bench_dirs = _find_benchmark_dirs(root)
    if not bench_dirs:
        raise SystemExit(f"No benchmark directories found under: {root}")

    run_frames: list[pd.DataFrame] = []
    cp_frames: list[pd.DataFrame] = []
    checkpoint_gens: set[int] = set()
    for bench in bench_dirs:
        runs_csv = bench / "runs.csv"
        if runs_csv.exists():
            df = pd.read_csv(runs_csv)
            df.insert(0, "bench_dir", str(bench.resolve()))
            run_frames.append(df)

        cp_csv = bench / "checkpoint_runs.csv"
        if cp_csv.exists():
            cdf = pd.read_csv(cp_csv)
            cdf.insert(0, "bench_dir", str(bench.resolve()))
            cp_frames.append(cdf)

        cfg_path = bench / "config.json"
        if cfg_path.exists():
            try:
                payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                for g in payload.get("checkpoint_gens", []) or []:
                    checkpoint_gens.add(int(g))
            except Exception:
                pass

    if not run_frames:
        raise SystemExit(f"No runs.csv files found under: {root}")

    runs_df = pd.concat(run_frames, ignore_index=True)
    runs_csv_out = outdir / "runs_combined.csv"
    runs_df.to_csv(runs_csv_out, index=False)

    def _safe_col(name: str) -> list[float]:
        if name not in runs_df.columns:
            return []
        return [float(v) for v in runs_df[name].tolist() if pd.notna(v)]

    runs_summary = {
        "schema_version": 1,
        "root": str(root),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bench_dirs": [str(p.resolve()) for p in bench_dirs],
        "runs_csv": str(runs_csv_out),
        "n_runs": int(len(runs_df)),
        "best_backtest_sharpe": _aggregate(_safe_col("best_backtest_sharpe")),
        "ensemble_portfolio_sharpe": _aggregate(_safe_col("ensemble_portfolio_sharpe")),
        "selected_avg_abs_corr": _aggregate(_safe_col("selected_avg_abs_corr")),
        "selected_max_abs_corr": _aggregate(_safe_col("selected_max_abs_corr")),
        "raw_topk_avg_abs_corr": _aggregate(_safe_col("raw_topk_avg_abs_corr")),
        "raw_topk_max_abs_corr": _aggregate(_safe_col("raw_topk_max_abs_corr")),
        "time_to_threshold_gen": _aggregate(_safe_col("time_to_threshold_gen")),
    }
    runs_summary_path = outdir / "runs_summary.json"
    runs_summary_path.write_text(json.dumps(runs_summary, indent=2), encoding="utf-8")

    if cp_frames:
        cp_df = pd.concat(cp_frames, ignore_index=True)
        cp_csv_out = outdir / "checkpoint_runs_combined.csv"
        cp_df.to_csv(cp_csv_out, index=False)
        per_gen: dict[str, Any] = {}
        for g in sorted(int(x) for x in cp_df["generation"].dropna().unique().tolist()):
            gdf = cp_df[cp_df["generation"] == g]
            per_gen[f"gen_{g:03d}"] = {
                "n": int(len(gdf)),
                "best_backtest_sharpe": _aggregate(
                    [float(v) for v in gdf["best_backtest_sharpe"].tolist() if pd.notna(v)]
                ),
                "ensemble_portfolio_sharpe": _aggregate(
                    [float(v) for v in gdf["ensemble_portfolio_sharpe"].tolist() if pd.notna(v)]
                ),
                "selected_avg_abs_corr": _aggregate(
                    [float(v) for v in gdf["selected_avg_abs_corr"].tolist() if pd.notna(v)]
                ),
            }
        cp_summary = {
            "schema_version": 1,
            "root": str(root),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_gens": sorted(checkpoint_gens),
            "checkpoint_runs_csv": str(cp_csv_out),
            "summary_by_generation": per_gen,
            "pairwise_scientific": _checkpoint_pairwise_report(cp_df),
        }
        cp_summary_path = outdir / "checkpoint_summary_combined.json"
        cp_summary_path.write_text(json.dumps(cp_summary, indent=2), encoding="utf-8")
        print(f"[aggregate] Wrote checkpoint summary -> {cp_summary_path}")

    print(f"[aggregate] Wrote runs CSV -> {runs_csv_out}")
    print(f"[aggregate] Wrote runs summary -> {runs_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
