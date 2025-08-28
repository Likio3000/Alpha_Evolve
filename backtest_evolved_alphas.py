from __future__ import annotations
import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import fields as dc_fields

import numpy as np
import pandas as pd

from config import BacktestConfig
from utils.data_loading_common import DataLoadError
from utils.logging_setup import setup_logging
from utils.errors import BacktestError
from utils.cli import add_dataclass_args
from utils.config_layering import (
    load_config_file,
    layer_dataclass_config,
    _flatten_sectioned_config,
)  # type: ignore
from alpha_framework import (
    AlphaProgram,
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    SCALAR_FEATURE_NAMES,
    OP_REGISTRY,
)
from backtesting_components import (
    load_and_align_data_for_backtest,
    backtest_cross_sectional_alpha,
)


def _derive_state_vars(prog: AlphaProgram) -> Dict[str, str]:
    """Infer required state variables from the program structure."""
    feature_vars = set(SCALAR_FEATURE_NAMES) | set(CROSS_SECTIONAL_FEATURE_VECTOR_NAMES)
    defined = set(feature_vars)
    state_vars: Dict[str, str] = {}

    for op in prog.setup + prog.predict_ops + prog.update_ops:
        spec = OP_REGISTRY[op.opcode]
        for idx, inp in enumerate(op.inputs):
            if inp not in defined and inp not in state_vars:
                arg_type = spec.in_types[idx]
                if spec.is_elementwise and arg_type == "scalar":
                    arg_type = "vector"
                state_vars[inp] = arg_type
        defined.add(op.out)

    # Always include defaults expected during evolution
    state_vars.setdefault("prev_s1_vec", "vector")
    state_vars.setdefault("rolling_mean_custom", "vector")
    return state_vars


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
def load_programs_from_pickle(
    n_to_load: int, pickle_filepath: str
) -> List[Tuple[AlphaProgram, float]]:
    """Load the top evolved programs from a pickle file.

    Parameters
    ----------
    n_to_load:
        Number of program/score pairs to return.
    pickle_filepath:
        Path to the pickle file produced by the evolution stage.

    Returns
    -------
    list of tuple[AlphaProgram, float]
        Up to ``n_to_load`` ``(program, metric)`` pairs from the pickle.

    Raises
    ------
    BacktestError
        If the pickle file is missing or cannot be deserialised.
    """

    if not Path(pickle_filepath).exists():
        raise BacktestError(f"Pickle file not found: {pickle_filepath}")
    try:
        with open(pickle_filepath, "rb") as fh:
            data: List[Tuple[AlphaProgram, float]] = pickle.load(fh)
        return data[:n_to_load]
    except Exception as e:
        raise BacktestError(f"Error loading pickle {pickle_filepath}: {e}")


# --------------------------------------------------------------------------- #
#  CLI → BacktestConfig                                                       #
# --------------------------------------------------------------------------- #
def parse_args() -> tuple[BacktestConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description="Back-test evolved cross-sectional alphas")

    # ­­­ file / misc (stay as raw CLI params) ­­­ #
    p.add_argument(
        "--input",
        default="evolved_top_alphas.pkl",
        help="Pickle produced by the evolution stage",
    )
    p.add_argument(
        "--outdir",
        default="evolved_bt_cs_results",
        help="Directory to write CSV summaries",
    )
    p.add_argument("--debug_prints", action="store_true")
    p.add_argument("--annualization_factor_override", type=float, default=None)
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    p.add_argument("--log-file", default=None, help="Optional log file path")
    p.add_argument(
        "--config",
        default=None,
        help="Optional TOML/YAML config file (file < env < CLI)",
    )
    # Cache controls (mirror pipeline)
    p.add_argument(
        "--disable-align-cache",
        action="store_true",
        help="Disable alignment cache (AE_DISABLE_ALIGN_CACHE=1)",
    )
    p.add_argument(
        "--align-cache-dir",
        default=None,
        help="Custom directory for alignment cache (AE_ALIGN_CACHE_DIR)",
    )

    # BacktestConfig flags auto-added from dataclass (normalized only; no aliases)
    choices_map = {
        "scale": ["zscore", "rank", "sign", "madz", "winsor"],
        "max_lookback_data_option": [
            "common_1200",
            "specific_long_10k",
            "full_overlap",
        ],
    }
    add_dataclass_args(p, BacktestConfig, choices_map=choices_map)

    ns = p.parse_args()

    # feed only recognised fields into the dataclass (including inherited)
    bt_field_names = {f.name for f in dc_fields(BacktestConfig)}
    cli_kwargs = {k: v for k, v in vars(ns).items() if k in bt_field_names}

    # load config file (if any) and extract backtest section or flat
    file_cfg: dict | None = None
    if getattr(ns, "config", None):
        raw = load_config_file(ns.config)
        # backtest section preferred; common synonyms
        for section in ("backtest", "bt"):
            if section in raw:
                file_cfg = _flatten_sectioned_config(raw, section)
                break
        if file_cfg is None:
            file_cfg = _flatten_sectioned_config(raw, None)

    # env precedence: common then BT-specific
    merged = layer_dataclass_config(
        BacktestConfig,
        file_cfg=file_cfg,
        env_prefixes=("AE_", "AE_BT_"),
        cli_overrides=cli_kwargs,
    )
    cfg = BacktestConfig(**merged)

    return cfg, ns


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def _validate_config(cfg: BacktestConfig) -> None:
    if cfg.top_to_backtest <= 0:
        raise ValueError("--top must be > 0")
    if cfg.hold < 1:
        raise ValueError("--hold must be >= 1")
    if cfg.fee < 0:
        raise ValueError("--fee must be >= 0")
    if cfg.long_short_n < 0:
        raise ValueError("--long_short_n must be >= 0")
    if cfg.annualization_factor <= 0:
        raise ValueError("--annualization_factor must be > 0")
    # Cross-module constraint: intrabar stop logic requires eval_lag == 1
    if cfg.stop_loss_pct and cfg.stop_loss_pct > 0.0 and cfg.eval_lag != 1:
        raise ValueError("--stop_loss_pct requires --eval_lag 1")


def run(
    cfg: BacktestConfig,
    *,
    outdir: Path,
    programs_pickle: Path,
    debug_prints: bool = False,
    annualization_factor_override: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Run the back-test programmatically.

    Returns the path to the CSV summary.
    """
    _validate_config(cfg)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lg = logger or logging.getLogger(__name__)

    # Seed for reproducibility
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        lg.info("Using seed %s for back-test reproducibility.", cfg.seed)

    # Load and align data
    lg.info("Loading data from '%s' …", cfg.data_dir)
    try:
        aligned_dfs, common_index, stock_symbols = load_and_align_data_for_backtest(
            cfg.data_dir,
            cfg.max_lookback_data_option,
            cfg.min_common_points,
            cfg.eval_lag,
        )
    except DataLoadError as e:
        lg.error("Backtest data loading failed: %s", e)
        raise
    lg.info(
        "%d symbols | %d bars (%s → %s)",
        len(stock_symbols),
        len(common_index),
        getattr(common_index, "min", lambda: "?")(),
        getattr(common_index, "max", lambda: "?")(),
    )

    # Load programs from provided pickle
    programs = load_programs_from_pickle(cfg.top_to_backtest, str(programs_pickle))
    if not programs:
        raise ValueError("Nothing to back-test – pickle empty or --top 0?")

    results: List[Dict[str, Any]] = []
    # Keep per-alpha net return series for optional ensemble building
    per_alpha_returns: List[tuple[str, List[float]]] = []
    for idx, (prog, evo_ic) in enumerate(programs, 1):
        lg.info("Back-testing alpha #%02d (evo IC %+0.4f)", idx, evo_ic)
        lg.debug("Program: %s", prog.to_string(max_len=200))

        metrics = backtest_cross_sectional_alpha(
            prog=prog,
            aligned_dfs=aligned_dfs,
            common_time_index=common_index,
            stock_symbols=stock_symbols,
            n_stocks=len(stock_symbols),
            fee_bps=cfg.fee,
            lag=cfg.eval_lag,
            hold=cfg.hold,
            long_short_n=cfg.long_short_n,
            scale_method=cfg.scale,
            winsor_p=cfg.winsor_p,
            initial_state_vars_config=_derive_state_vars(prog),
            scalar_feature_names=SCALAR_FEATURE_NAMES,
            cross_sectional_feature_vector_names=CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
            debug_prints=debug_prints,
            annualization_factor=(
                annualization_factor_override
                if annualization_factor_override is not None
                else cfg.annualization_factor
            ),
            stop_loss_pct=cfg.stop_loss_pct,
            sector_neutralize_positions=cfg.sector_neutralize_positions,
            volatility_target=cfg.volatility_target,
            volatility_lookback=cfg.volatility_lookback,
            max_leverage=cfg.max_leverage,
            min_leverage=cfg.min_leverage,
            dd_limit=cfg.dd_limit,
            dd_reduction=cfg.dd_reduction,
        )

        # Persist per-bar diagnostics for plotting
        try:
            ts_len = len(metrics.get("RetNet", []))
            if ts_len > 0:
                dates = list(common_index[:ts_len])
                import pandas as _pd

                df_ts = _pd.DataFrame(
                    {
                        "date": dates,
                        "ret_net": metrics.get("RetNet"),
                        "equity": metrics.get("EquityCurve"),
                        "exposure_mult": metrics.get("ExposureMult"),
                        "drawdown": metrics.get("Drawdown"),
                        "stop_hits": metrics.get("StopHitsPerBar"),
                    }
                )
                ts_path = outdir / f"alpha_{idx:02d}_timeseries.csv"
                df_ts.to_csv(ts_path, index=False)
                metrics["TimeseriesFile"] = str(ts_path)
                metrics["TS"] = ts_path.name  # short, tidy link label for tables
                # Save returns for ensemble selection before dropping arrays
                try:
                    per_alpha_returns.append((f"Alpha_{idx:02d}", list(metrics.get("RetNet", []))))
                except Exception:
                    pass
                # Drop bulky arrays from summary entry
                for k in (
                    "RetNet",
                    "EquityCurve",
                    "ExposureMult",
                    "Drawdown",
                    "StopHitsPerBar",
                ):
                    metrics.pop(k, None)
        except Exception:
            pass

        metrics["Ops"] = prog.size
        metrics.update(
            {
                "AlphaID": f"Alpha_{idx:02d}",
                "OriginalMetric": evo_ic,
                "Program": prog.to_string(max_len=1_000_000_000),
            }
        )
        results.append(metrics)

        lg.info(
            " └─ Sharpe %+0.3f  AnnRet %6.2f%%  MaxDD %6.2f%%  Turnover %.4f",
            metrics.get("Sharpe", 0.0),
            metrics.get("AnnReturn", 0.0) * 100,
            metrics.get("MaxDD", 0.0) * 100,
            metrics.get("Turnover", 0.0),
        )

    # Save summaries
    if results:
        df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
        actual_n = len(programs)
        summary_csv = outdir / f"backtest_summary_top{actual_n}.csv"
        df.to_csv(summary_csv, index=False, float_format="%.4f")
        # JSON alongside
        summary_json = outdir / f"backtest_summary_top{actual_n}.json"
        with open(summary_json, "w") as fh:
            json.dump(results, fh, indent=2)
        # Optional ensemble: greedily pick low-corr subset and compute portfolio metrics
        try:
            ens_n = int(getattr(cfg, "ensemble_size", 0) or 0)
            ens_mode = bool(getattr(cfg, "ensemble_mode", False))
            if ens_mode or ens_n > 0:
                import numpy as _np
                # Build matrix of aligned returns (truncate to min length)
                if per_alpha_returns:
                    min_len = min(len(r) for _, r in per_alpha_returns)
                    names = [name for name, _ in per_alpha_returns]
                    R = _np.array([_np.array(r[:min_len], dtype=float) for _, r in per_alpha_returns])  # shape (K, T)
                    # Rank alphas by Sharpe for seed ordering
                    name_to_sharpe = {row["AlphaID"]: float(row.get("Sharpe", 0.0)) for _, row in df.iterrows()}
                    order = sorted(range(len(names)), key=lambda i: name_to_sharpe.get(names[i], 0.0), reverse=True)
                    # Greedy selection under max corr constraint
                    max_corr = float(getattr(cfg, "ensemble_max_corr", 0.3))
                    selected: List[int] = []
                    target_k = ens_n if ens_n > 0 else len(names)
                    for i in order:
                        if len(selected) >= target_k:
                            break
                        ok = True
                        for j in selected:
                            c = float(_np.corrcoef(R[i], R[j])[0, 1]) if min_len > 1 else 0.0
                            if _np.isnan(c) or abs(c) > max_corr:
                                ok = False
                                break
                        if ok:
                            selected.append(i)
                    # Fallback: ensure at least top-2 if constraint too strict
                    if not selected and len(order) >= 1:
                        selected = order[: min(2, len(order))]
                    if selected:
                        port_ret = _np.mean(R[selected, :], axis=0)
                        # Compute portfolio metrics (mirror core_logic summary)
                        eq = _np.cumprod(1 + port_ret)
                        peak = _np.maximum.accumulate(eq)
                        dd = (eq - peak) / (peak + 1e-9)
                        mean_ret = float(_np.mean(port_ret)) if port_ret.size else 0.0
                        std_ret = float(_np.std(port_ret, ddof=0)) if port_ret.size else 0.0
                        ann = float(cfg.annualization_factor)
                        sharpe = (mean_ret / (std_ret + 1e-9)) * (_np.sqrt(ann) if ann and ann > 0 else 1.0)
                        total_return = eq[-1] - 1.0 if eq.size else 0.0
                        years = (port_ret.size / ann) if ann and ann > 0 else 1.0
                        ann_ret = ((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0
                        ann_vol = std_ret * (_np.sqrt(ann) if ann and ann > 0 else 1.0)
                        max_dd = float(-_np.min(dd)) if dd.size else 0.0
                        ens_rows = [{
                            "Sharpe": sharpe,
                            "AnnReturn": ann_ret,
                            "AnnVol": ann_vol,
                            "MaxDD": max_dd,
                            "Members": [names[i] for i in selected],
                            "K": len(selected),
                        }]
                        ens_csv = outdir / "backtest_summary_ensemble.csv"
                        _pd.DataFrame(ens_rows).to_csv(ens_csv, index=False)
                        # Save ensemble timeseries
                        ens_ts = _pd.DataFrame({
                            "date": list(common_index[:min_len]),
                            "ret_net": port_ret,
                            "equity": eq,
                            "drawdown": dd,
                        })
                        ens_ts.to_csv(outdir / "ensemble_timeseries.csv", index=False)
                        lg.info("Ensemble (K=%d) Sharpe %+0.3f AnnRet %6.2f%% MaxDD %6.2f%%", len(selected), sharpe, ann_ret * 100, max_dd * 100)
        except Exception:
            # Ensemble is best-effort; do not fail the run if issues arise
            pass
        try:
            printable = df.drop(columns=["Program", "TimeseriesFile"], errors="ignore")
            lg.info("\n%s", printable.to_string(index=False))
        except Exception:
            pass
        lg.info("Back-test summary written → %s", summary_csv)
        return summary_csv
    else:
        raise RuntimeError("Backtest produced no results")


def main() -> None:
    cfg, cli = parse_args()

    # Apply cache controls early so loaders honor them
    try:
        import os

        if getattr(cli, "disable_align_cache", False):
            os.environ["AE_DISABLE_ALIGN_CACHE"] = "1"
        if getattr(cli, "align_cache_dir", None):
            os.environ["AE_ALIGN_CACHE_DIR"] = str(cli.align_cache_dir)
    except Exception:
        pass

    # Prepare logging for standalone invocation (tqdm-friendly, colored)
    level = getattr(logging, str(cli.log_level).upper(), logging.INFO)
    setup_logging(level=level, log_file=cli.log_file)

    logger = logging.getLogger(__name__)

    try:
        _validate_config(cfg)
    except Exception as e:
        logger.error("Invalid backtest configuration: %s", e)
        sys.exit(2)

    outdir = Path(cli.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Seed for reproducibility
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        logger.info("Using seed %s for back-test reproducibility.", cfg.seed)

    # Load and align data
    logger.info("Loading data from '%s' …", cfg.data_dir)
    aligned_dfs, common_index, stock_symbols = load_and_align_data_for_backtest(
        cfg.data_dir,
        cfg.max_lookback_data_option,
        cfg.min_common_points,
        cfg.eval_lag,
    )
    logger.info(
        "%d symbols | %d bars (%s → %s)",
        len(stock_symbols),
        len(common_index),
        getattr(common_index, "min", lambda: "?")(),
        getattr(common_index, "max", lambda: "?")(),
    )

    # Load programs
    try:
        programs = load_programs_from_pickle(cfg.top_to_backtest, cli.input)
    except BacktestError as e:
        logger.error(str(e))
        sys.exit(2)
    if not programs:
        sys.exit("Nothing to back-test – pickle empty or --top 0?")

    results: List[Dict[str, Any]] = []
    for idx, (prog, evo_ic) in enumerate(programs, 1):
        logger.info("Back-testing alpha #%02d (evo IC %+0.4f)", idx, evo_ic)
        logger.debug("Program: %s", prog.to_string(max_len=200))

        metrics = backtest_cross_sectional_alpha(
            prog=prog,
            aligned_dfs=aligned_dfs,
            common_time_index=common_index,
            stock_symbols=stock_symbols,
            n_stocks=len(stock_symbols),
            fee_bps=cfg.fee,
            lag=cfg.eval_lag,
            hold=cfg.hold,
            long_short_n=cfg.long_short_n,
            scale_method=cfg.scale,
            winsor_p=cfg.winsor_p,
            initial_state_vars_config=_derive_state_vars(prog),
            scalar_feature_names=SCALAR_FEATURE_NAMES,
            cross_sectional_feature_vector_names=CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
            debug_prints=cli.debug_prints,
            annualization_factor=(
                cli.annualization_factor_override
                if cli.annualization_factor_override is not None
                else cfg.annualization_factor
            ),
            stop_loss_pct=cfg.stop_loss_pct,
            sector_neutralize_positions=cfg.sector_neutralize_positions,
            volatility_target=cfg.volatility_target,
            volatility_lookback=cfg.volatility_lookback,
            max_leverage=cfg.max_leverage,
            min_leverage=cfg.min_leverage,
            dd_limit=cfg.dd_limit,
            dd_reduction=cfg.dd_reduction,
        )

        # Persist per-bar diagnostics for plotting
        try:
            ts_len = len(metrics.get("RetNet", []))
            if ts_len > 0:
                dates = list(common_index[:ts_len])
                import pandas as _pd

                df_ts = _pd.DataFrame(
                    {
                        "date": dates,
                        "ret_net": metrics.get("RetNet"),
                        "equity": metrics.get("EquityCurve"),
                        "exposure_mult": metrics.get("ExposureMult"),
                        "drawdown": metrics.get("Drawdown"),
                        "stop_hits": metrics.get("StopHitsPerBar"),
                    }
                )
                ts_path = outdir / f"alpha_{idx:02d}_timeseries.csv"
                df_ts.to_csv(ts_path, index=False)
                metrics["TimeseriesFile"] = str(ts_path)
                for k in (
                    "RetNet",
                    "EquityCurve",
                    "ExposureMult",
                    "Drawdown",
                    "StopHitsPerBar",
                ):
                    metrics.pop(k, None)
        except Exception:
            pass

        metrics["Ops"] = prog.size
        metrics.update(
            {
                "AlphaID": f"Alpha_{idx:02d}",
                "OriginalMetric": evo_ic,
                "Program": prog.to_string(max_len=1_000_000_000),
            }
        )
        results.append(metrics)

        logger.info(
            " └─ Sharpe %+0.3f  AnnRet %6.2f%%  MaxDD %6.2f%%  Turnover %.4f",
            metrics.get("Sharpe", 0.0),
            metrics.get("AnnReturn", 0.0) * 100,
            metrics.get("MaxDD", 0.0) * 100,
            metrics.get("Turnover", 0.0),
        )

    # Save summaries
    if results:
        df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
        summary_csv = outdir / f"backtest_summary_top{cfg.top_to_backtest}.csv"
        df.to_csv(summary_csv, index=False, float_format="%.4f")
        # JSON alongside
        summary_json = outdir / f"backtest_summary_top{cfg.top_to_backtest}.json"
        with open(summary_json, "w") as fh:
            json.dump(results, fh, indent=2)
        # Print compact table to stdout for human inspection
        try:
            printable = df.drop(columns=["Program", "TimeseriesFile"], errors="ignore")
            logger.info("\n%s", printable.to_string(index=False))
        except Exception:
            pass
        logger.info("Back-test summary written → %s", summary_csv)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
