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
    for cand in bt_dir.glob("backtest_summary_top*.csv"):
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

    # Optional overrides (kept small; prefer config files for most tuning)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--pop-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--top-to-backtest", type=int, default=None)
    p.add_argument("--ensemble-size", type=int, default=None)
    p.add_argument("--ensemble-max-corr", type=float, default=None)
    p.add_argument("--ensemble-corr-lambda", type=float, default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config) if args.config else (
        ROOT / ("configs/bench_sp500_small_quick.toml" if args.mode == "quick" else "configs/bench_sp500_full.toml")
    )
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    base_evo, base_bt = _load_configs(config_path)

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

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = (ROOT / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir).resolve()
    bench_dir = out_root / f"sp500_bench_{args.mode}_{stamp}"
    runs_root = bench_dir / "pipeline_runs_cs"
    bench_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    (bench_dir / "config.json").write_text(json.dumps({"path": str(config_path), "evolution": asdict(base_evo), "backtest": asdict(base_bt)}, indent=2))

    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
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
        )
        t0 = time.perf_counter()
        run_dir = run_pipeline_programmatic(evo, bt, opts)
        elapsed = time.perf_counter() - t0

        record = {
            "seed": int(seed),
            "elapsed_sec": float(elapsed),
        }
        record.update(_collect_run_summary(Path(run_dir), threshold_sharpe=float(args.threshold_sharpe)))
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
    print(f"[benchmark] Wrote reports -> {bench_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
