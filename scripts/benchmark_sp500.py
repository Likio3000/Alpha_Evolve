#!/usr/bin/env python3
"""Reproducible benchmark harness for the SP500 pipeline.

Quick mode is sandbox/CI-friendly (uses data_sp500_small). Full mode targets the
daily SP500 dataset and is intended for a normal machine.

Outputs machine-readable CSV/JSON summaries for easy diffs.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha_evolve.cli.pipeline import PipelineOptions, run_pipeline_programmatic
from alpha_evolve.backtesting import engine as bt_engine
from alpha_evolve.config import BacktestConfig, EvolutionConfig
from alpha_evolve.config.layering import (
    _flatten_sectioned_config,
    layer_dataclass_config,
    load_config_file,
)


def _parse_seeds(spec: str) -> list[int]:
    s = (spec or "").strip()
    if not s:
        return []
    if ":" in s:
        a, b = s.split(":", 1)
        start = int(a.strip() or "0")
        end = int(b.strip())
        if end < start:
            raise ValueError("seed range must be start:end with end>=start")
        return list(range(start, end))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


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
    return out


def _load_configs(path: Path) -> tuple[EvolutionConfig, BacktestConfig]:
    raw = load_config_file(str(path))
    evo_file_cfg = _flatten_sectioned_config(raw, "evolution") if "evolution" in raw else None
    bt_file_cfg = _flatten_sectioned_config(raw, "backtest") if "backtest" in raw else None
    if evo_file_cfg is None and bt_file_cfg is None:
        flat = _flatten_sectioned_config(raw, None)
        evo_file_cfg = flat
        bt_file_cfg = flat

    evo_kwargs = layer_dataclass_config(EvolutionConfig, file_cfg=evo_file_cfg, env_prefixes=(), cli_overrides={})
    bt_kwargs = layer_dataclass_config(BacktestConfig, file_cfg=bt_file_cfg, env_prefixes=(), cli_overrides={})
    return EvolutionConfig(**evo_kwargs), BacktestConfig(**bt_kwargs)


def _pairwise_corr_stats(corr: pd.DataFrame, members: Sequence[str]) -> dict[str, Any]:
    members = [m for m in members if m in corr.index]
    if len(members) < 2:
        return {"k": len(members), "avg_abs_corr": 0.0, "max_abs_corr": 0.0}
    mat = corr.loc[members, members].to_numpy(dtype=float)
    mat = abs(mat)
    # extract upper triangle off-diagonal
    vals = mat[np.triu_indices(mat.shape[0], k=1)]
    vals = vals[pd.notna(vals)]
    if vals.size == 0:
        return {"k": len(members), "avg_abs_corr": 0.0, "max_abs_corr": 0.0}
    return {
        "k": int(len(members)),
        "avg_abs_corr": float(vals.mean()),
        "max_abs_corr": float(vals.max()),
    }


def _time_to_threshold(diags: list[dict[str, Any]], threshold: float) -> int | None:
    for entry in diags:
        try:
            gen = int(entry.get("generation", 0))
            best = entry.get("best") or {}
            s = float(best.get("sharpe", float("nan")))
        except Exception:
            continue
        if gen > 0 and s >= threshold:
            return gen
    return None


def _collect_run_summary(run_dir: Path, *, threshold_sharpe: float) -> dict[str, Any]:
    summary_path = run_dir / "SUMMARY.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    bt_dir = run_dir / "backtest_portfolio_csvs"
    best_sharpe = None
    best_alpha = None
    try:
        best_metrics = summary.get("best_metrics") or {}
        best_sharpe = float(best_metrics.get("Sharpe")) if "Sharpe" in best_metrics else None
        best_alpha = str(best_metrics.get("AlphaID")) if best_metrics.get("AlphaID") is not None else None
    except Exception:
        best_sharpe = None
        best_alpha = None

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

    raw_top_members: list[str] = []
    summary_csv = None
    for cand in sorted(bt_dir.glob("backtest_summary_top*.csv")):
        summary_csv = cand
        break
    if summary_csv and summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            df = df.sort_values("Sharpe", ascending=False)
            k = len(ens_members) if ens_members else min(5, int(len(df)))
            raw_top_members = [str(x) for x in df.head(k).get("AlphaID", []).tolist() if x]
        except Exception:
            raw_top_members = []

    corr_stats_ens = _pairwise_corr_stats(corr_df, ens_members) if corr_df is not None else None
    corr_stats_raw = _pairwise_corr_stats(corr_df, raw_top_members) if corr_df is not None else None

    diags_path = run_dir / "diagnostics.json"
    time_to_thr = None
    if diags_path.exists():
        try:
            diags = json.loads(diags_path.read_text(encoding="utf-8"))
            if isinstance(diags, list):
                time_to_thr = _time_to_threshold(diags, threshold_sharpe)
        except Exception:
            time_to_thr = None

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
        "run_dir": str(run_dir),
        "best_backtest_sharpe": best_sharpe,
        "best_alpha": best_alpha,
        "ensemble_members": ens_members,
        "ensemble_portfolio_sharpe": ens_port_sharpe,
        "raw_top_members": raw_top_members,
        "corr_selected": corr_stats_ens,
        "corr_raw_topk": corr_stats_raw,
        "time_to_threshold_gen": time_to_thr,
    }


def _collect_backtest_dir_summary(bt_dir: Path) -> dict[str, Any]:
    """Collect summary metrics from a backtest output directory."""
    best_sharpe = None
    best_alpha = None
    raw_top_members: list[str] = []
    summary_csv = None
    for cand in sorted(bt_dir.glob("backtest_summary_top*.csv")):
        summary_csv = cand
        break
    if summary_csv and summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            df = df.sort_values("Sharpe", ascending=False)
            if len(df) > 0 and "Sharpe" in df.columns:
                best_sharpe = float(df.iloc[0]["Sharpe"])
                best_alpha = str(df.iloc[0].get("AlphaID", ""))
            raw_top_members = [
                str(x) for x in df.head(min(5, int(len(df)))).get("AlphaID", []).tolist() if x
            ]
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


def _run_checkpoint_backtests(
    run_dir: Path,
    *,
    base_bt: BacktestConfig,
    checkpoint_gens: Sequence[int],
    debug_prints: bool,
    logger: logging.Logger,
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return out

    for g in sorted({int(x) for x in checkpoint_gens if int(x) > 0}):
        pkl_path = ckpt_root / f"hof_gen_{g:03d}.pkl"
        if not pkl_path.exists():
            continue
        bt_cfg = copy.deepcopy(base_bt)
        bt_outdir = run_dir / "checkpoint_backtests" / f"gen_{g:03d}"
        bt_outdir.mkdir(parents=True, exist_ok=True)
        try:
            bt_engine.run(
                bt_cfg,
                outdir=bt_outdir,
                programs_pickle=pkl_path,
                debug_prints=debug_prints,
                annualization_factor_override=None,
                logger=logger,
            )
        except Exception:
            continue
        rec = {"generation": int(g), "programs_pickle": str(pkl_path), "bt_outdir": str(bt_outdir)}
        rec.update(_collect_backtest_dir_summary(bt_outdir))
        out[int(g)] = rec
    return out


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
    ge_one = int(np.sum(means >= obs))
    ge_two = int(np.sum(np.abs(means) >= abs(obs)))
    p_one = float((ge_one + 1) / (n_perm + 1))
    p_two = float((ge_two + 1) / (n_perm + 1))
    return p_one, p_two


def _holm_bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(
        enumerate(float(p) for p in p_values),
        key=lambda t: t[1],
    )
    adjusted = [1.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        factor = n - rank + 1
        padj = min(1.0, max(0.0, p) * factor)
        running_max = max(running_max, padj)
        adjusted[orig_idx] = running_max
    return adjusted


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
    pending_adjustment: list[dict[str, Any]] = []
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
                    "scientific_pass": bool(
                        np.isfinite(ci_lo) and ci_lo > 0 and p_one <= 0.05
                    ),
                }
            )
            pending_adjustment.append(pair_payload["metrics"][-1])
        pairs.append(pair_payload)
    if pending_adjustment:
        adj = _holm_bonferroni_adjust(
            [float(m.get("p_perm_one_sided", 1.0)) for m in pending_adjustment]
        )
        for m, p_adj in zip(pending_adjustment, adj):
            ci = m.get("ci95_mean_improvement") or [float("nan"), float("nan")]
            ci_lo = float(ci[0]) if isinstance(ci, list) and ci else float("nan")
            raw_pass = bool(m.get("scientific_pass", False))
            m["scientific_pass_unadjusted"] = raw_pass
            m["p_perm_one_sided_holm"] = float(p_adj)
            m["scientific_pass"] = bool(
                np.isfinite(ci_lo) and ci_lo > 0 and float(p_adj) <= 0.05
            )
    return {"schema_version": 1, "pairs": pairs}


def _aggregate(values: list[float]) -> dict[str, Any]:
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


def _flatten_record_for_csv(record: Mapping[str, Any]) -> dict[str, Any]:
    corr_sel = record.get("corr_selected") or {}
    corr_raw = record.get("corr_raw_topk") or {}
    row: dict[str, Any] = {
        "seed": record.get("seed"),
        "elapsed_sec": record.get("elapsed_sec"),
        "run_dir": record.get("run_dir"),
        "best_alpha": record.get("best_alpha"),
        "best_backtest_sharpe": record.get("best_backtest_sharpe"),
        "ensemble_portfolio_sharpe": record.get("ensemble_portfolio_sharpe"),
        "time_to_threshold_gen": record.get("time_to_threshold_gen"),
        "ensemble_k": corr_sel.get("k"),
        "selected_avg_abs_corr": corr_sel.get("avg_abs_corr"),
        "selected_max_abs_corr": corr_sel.get("max_abs_corr"),
        "raw_topk_k": corr_raw.get("k"),
        "raw_topk_avg_abs_corr": corr_raw.get("avg_abs_corr"),
        "raw_topk_max_abs_corr": corr_raw.get("max_abs_corr"),
    }
    try:
        raw_avg = float(row["raw_topk_avg_abs_corr"]) if row["raw_topk_avg_abs_corr"] is not None else None
        sel_avg = float(row["selected_avg_abs_corr"]) if row["selected_avg_abs_corr"] is not None else None
        row["delta_avg_abs_corr_selected_minus_raw"] = None if raw_avg is None or sel_avg is None else (sel_avg - raw_avg)
    except Exception:
        row["delta_avg_abs_corr_selected_minus_raw"] = None
    return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SP500 benchmark harness (quick/full)")
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--config", default=None, help="Override TOML config path")
    p.add_argument("--seeds", default=None, help="Seed list: '0,1,2' or range '0:5' (end exclusive)")
    p.add_argument("--threshold-sharpe", type=float, default=1.0, help="Sharpe proxy threshold for time-to-threshold")
    p.add_argument("--outdir", default="artifacts/benchmarks", help="Directory to write benchmark reports")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Disable diagnostics/backtest plotting to speed up benchmark throughput",
    )

    # Optional overrides (kept small; prefer config files for most tuning)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--pop-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--top-to-backtest", type=int, default=None)
    p.add_argument("--ensemble-size", type=int, default=None)
    p.add_argument("--ensemble-max-corr", type=float, default=None)
    p.add_argument("--ensemble-corr-lambda", type=float, default=None)
    p.add_argument(
        "--ensemble-refine-swaps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable local-swap refinement after greedy ensemble selection",
    )
    p.add_argument("--ensemble-refine-max-passes", type=int, default=None)
    p.add_argument(
        "--checkpoint-gens",
        default=None,
        help="Comma-separated generations to checkpoint and backtest from the same run trajectory (e.g. '45,60,90')",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logger = logging.getLogger("benchmark_sp500")

    config_path = Path(args.config) if args.config else (
        ROOT / ("configs/bench_sp500_small_quick.toml" if args.mode == "quick" else "configs/bench_sp500_full.toml")
    )
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    base_evo, base_bt = _load_configs(config_path)
    checkpoint_source = (
        _parse_int_list(args.checkpoint_gens)
        if args.checkpoint_gens is not None
        else [int(x) for x in (getattr(base_evo, "checkpoint_gens", ()) or ())]
    )
    checkpoint_gens = sorted({int(x) for x in checkpoint_source if int(x) > 0})
    if checkpoint_gens:
        max_ckpt = max(checkpoint_gens)
        if args.generations is not None and int(args.generations) < max_ckpt:
            raise SystemExit(
                f"--generations ({args.generations}) must be >= max checkpoint generation ({max_ckpt})"
            )
        base_evo.generations = max(int(base_evo.generations), int(max_ckpt))
        base_evo.checkpoint_gens = tuple(checkpoint_gens)

    seeds = _parse_seeds(args.seeds) if args.seeds else ([0, 1] if args.mode == "quick" else [0, 1, 2, 3, 4])
    if not seeds:
        raise SystemExit("No seeds provided")

    # Apply lightweight overrides.
    if args.generations is not None:
        base_evo.generations = int(args.generations)
    if args.pop_size is not None:
        base_evo.pop_size = int(args.pop_size)
    if args.workers is not None:
        base_evo.workers = int(args.workers)
    if args.top_to_backtest is not None:
        base_bt.top_to_backtest = int(args.top_to_backtest)
    if args.ensemble_size is not None:
        base_bt.ensemble_size = int(args.ensemble_size)
        base_bt.ensemble_mode = base_bt.ensemble_size > 0
    if args.ensemble_max_corr is not None:
        base_bt.ensemble_max_corr = float(args.ensemble_max_corr)
    if args.ensemble_corr_lambda is not None:
        base_bt.ensemble_corr_lambda = float(args.ensemble_corr_lambda)
    if args.ensemble_refine_swaps is not None:
        base_bt.ensemble_refine_swaps = bool(args.ensemble_refine_swaps)
    if args.ensemble_refine_max_passes is not None:
        base_bt.ensemble_refine_max_passes = int(args.ensemble_refine_max_passes)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = (ROOT / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir).resolve()
    bench_dir = out_root / f"sp500_bench_{args.mode}_{stamp}"
    runs_root = bench_dir / "pipeline_runs_cs"
    bench_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    (bench_dir / "config.json").write_text(
        json.dumps(
            {
                "path": str(config_path),
                "evolution": asdict(base_evo),
                "backtest": asdict(base_bt),
                "checkpoint_gens": checkpoint_gens,
            },
            indent=2,
        )
    )

    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for seed in seeds:
        evo = copy.deepcopy(base_evo)
        bt = copy.deepcopy(base_bt)
        evo.seed = int(seed)
        bt.seed = int(seed)
        # Keep quick mode sandbox-friendly.
        if args.mode == "quick":
            evo.workers = 1

        opts = PipelineOptions(
            debug_prints=bool(args.debug),
            log_level=str(args.log_level),
            output_dir=str(runs_root),
            generate_diagnostics_plots=not bool(args.skip_plots),
            generate_backtest_plots=not bool(args.skip_plots),
        )
        t0 = time.perf_counter()
        run_dir = run_pipeline_programmatic(evo, bt, opts)
        elapsed = time.perf_counter() - t0

        record = {
            "seed": int(seed),
            "elapsed_sec": float(elapsed),
        }
        record.update(_collect_run_summary(Path(run_dir), threshold_sharpe=float(args.threshold_sharpe)))
        if checkpoint_gens:
            ckpt_records = _run_checkpoint_backtests(
                Path(run_dir),
                base_bt=bt,
                checkpoint_gens=checkpoint_gens,
                debug_prints=bool(args.debug),
                logger=logger,
            )
            record["checkpoint_results"] = ckpt_records
            for g, ck in sorted(ckpt_records.items()):
                row = {
                    "seed": int(seed),
                    "generation": int(g),
                    "run_dir": str(run_dir),
                    "best_alpha": ck.get("best_alpha"),
                    "best_backtest_sharpe": ck.get("best_backtest_sharpe"),
                    "ensemble_portfolio_sharpe": ck.get("ensemble_portfolio_sharpe"),
                }
                corr_sel = ck.get("corr_selected") or {}
                corr_raw = ck.get("corr_raw_topk") or {}
                row["ensemble_k"] = corr_sel.get("k")
                row["selected_avg_abs_corr"] = corr_sel.get("avg_abs_corr")
                row["selected_max_abs_corr"] = corr_sel.get("max_abs_corr")
                row["raw_topk_k"] = corr_raw.get("k")
                row["raw_topk_avg_abs_corr"] = corr_raw.get("avg_abs_corr")
                row["raw_topk_max_abs_corr"] = corr_raw.get("max_abs_corr")
                checkpoint_rows.append(row)
        records.append(record)
        csv_rows.append(_flatten_record_for_csv(record))

    df = pd.DataFrame(csv_rows)
    df.to_csv(bench_dir / "runs.csv", index=False)
    (bench_dir / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    best_sharpes = [r.get("best_backtest_sharpe") for r in records if r.get("best_backtest_sharpe") is not None]
    time_to_thr = [float(r["time_to_threshold_gen"]) for r in records if r.get("time_to_threshold_gen") is not None]
    corr_sel_avg = [float((r.get("corr_selected") or {}).get("avg_abs_corr")) for r in records if r.get("corr_selected")]
    corr_sel_max = [float((r.get("corr_selected") or {}).get("max_abs_corr")) for r in records if r.get("corr_selected")]
    corr_raw_avg = [float((r.get("corr_raw_topk") or {}).get("avg_abs_corr")) for r in records if r.get("corr_raw_topk")]
    corr_raw_max = [float((r.get("corr_raw_topk") or {}).get("max_abs_corr")) for r in records if r.get("corr_raw_topk")]

    report = {
        "schema_version": 1,
        "mode": args.mode,
        "config": str(config_path),
        "seeds": seeds,
        "runs_dir": str(runs_root),
        "runs_csv": str(bench_dir / "runs.csv"),
        "best_backtest_sharpe": _aggregate([float(x) for x in best_sharpes]),
        "time_to_threshold_gen": _aggregate(time_to_thr),
        "selected_avg_abs_corr": _aggregate(corr_sel_avg),
        "selected_max_abs_corr": _aggregate(corr_sel_max),
        "raw_topk_avg_abs_corr": _aggregate(corr_raw_avg),
        "raw_topk_max_abs_corr": _aggregate(corr_raw_max),
    }
    (bench_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if checkpoint_rows:
        cp_df = pd.DataFrame(checkpoint_rows)
        cp_df.to_csv(bench_dir / "checkpoint_runs.csv", index=False)
        checkpoint_summary: dict[str, Any] = {}
        for g in sorted(cp_df["generation"].dropna().unique().tolist()):
            gdf = cp_df[cp_df["generation"] == g]
            checkpoint_summary[f"gen_{int(g):03d}"] = {
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
        cp_report = {
            "schema_version": 1,
            "checkpoint_gens": checkpoint_gens,
            "summary_by_generation": checkpoint_summary,
            "pairwise_scientific": _checkpoint_pairwise_report(checkpoint_rows),
        }
        (bench_dir / "checkpoint_summary.json").write_text(
            json.dumps(cp_report, indent=2),
            encoding="utf-8",
        )
    print(f"[benchmark] Wrote reports -> {bench_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
