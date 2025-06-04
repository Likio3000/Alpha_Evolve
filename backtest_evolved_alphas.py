from __future__ import annotations
import argparse
import pickle
import random
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from config import BacktestConfig                # ← NEW
from alpha_framework import (
    AlphaProgram,
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    SCALAR_FEATURE_NAMES,
)
from backtesting_components import (
    load_and_align_data_for_backtest,
    backtest_cross_sectional_alpha,
)

# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
def load_programs_from_pickle(n_to_load: int, pickle_filepath: str) \
        -> List[Tuple[AlphaProgram, float]]:
    if not Path(pickle_filepath).exists():
        sys.exit(f"Pickle file not found: {pickle_filepath}")
    try:
        with open(pickle_filepath, "rb") as fh:
            data: List[Tuple[AlphaProgram, float]] = pickle.load(fh)
        return data[:n_to_load]
    except Exception as e:
        sys.exit(f"Error loading pickle {pickle_filepath}: {e}")


# --------------------------------------------------------------------------- #
#  CLI → BacktestConfig                                                       #
# --------------------------------------------------------------------------- #
def parse_args() -> tuple[BacktestConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description="Back-test evolved cross-sectional alphas")

    # ­­­ file / misc (stay as raw CLI params) ­­­ #
    p.add_argument("--input", default="evolved_top_alphas.pkl",
                   help="Pickle produced by the evolution stage")
    p.add_argument("--outdir", default="evolved_bt_cs_results",
                   help="Directory to write CSV summaries")
    p.add_argument("--debug_prints", action="store_true")
    p.add_argument("--annualization_factor_override", type=float, default=None)

    # ­­­ back-test knobs – map straight into BacktestConfig ­­­ #
    p.add_argument("--top",               dest="top_to_backtest",     type=int,   default=argparse.SUPPRESS)
    p.add_argument("--data",              dest="data_dir",            default=argparse.SUPPRESS)
    p.add_argument("--fee",               type=float,                 default=argparse.SUPPRESS)
    p.add_argument("--hold",              type=int,                   default=argparse.SUPPRESS)
    p.add_argument("--scale",             choices=["zscore", "rank", "sign"],
                                                                  default=argparse.SUPPRESS)
    p.add_argument("--lag",               dest="eval_lag",            type=int,   default=argparse.SUPPRESS)
    p.add_argument("--data_alignment_strategy",
                   dest="max_lookback_data_option",
                   choices=["common_1200", "specific_long_10k", "full_overlap"],
                   default=argparse.SUPPRESS)
    p.add_argument("--min_common_data_points",
                   dest="min_common_points", type=int,                default=argparse.SUPPRESS)
    p.add_argument("--seed",              type=int,                   default=argparse.SUPPRESS)

    ns = p.parse_args()

    # feed only recognised fields into the dataclass
    cfg_kwargs = {k: v for k, v in vars(ns).items()
                  if k in BacktestConfig.__annotations__}
    cfg = BacktestConfig(**cfg_kwargs)

    return cfg, ns


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg, cli = parse_args()

    Path(cli.outdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ seed
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        print(f"Using seed {cfg.seed} for back-test reproducibility.")

    # ------------------------------------------------------------ load data
    print(f"Loading data from '{cfg.data_dir}' …")
    aligned_dfs, common_index, stock_symbols = load_and_align_data_for_backtest(
        cfg.data_dir,
        cfg.max_lookback_data_option,
        cfg.min_common_points,
    )
    print(f"{len(stock_symbols)} symbols | {len(common_index)} bars "
          f"({common_index.min()} → {common_index.max()})")

    # ----------------------------------------------------------- programmes
    programs = load_programs_from_pickle(cfg.top_to_backtest, cli.input)
    if not programs:
        sys.exit("Nothing to back-test – pickle empty or --top 0?")

    results: List[Dict[str, Any]] = []
    for idx, (prog, evo_ic) in enumerate(programs, 1):
        print(f"\nBack-testing alpha #{idx:02d}  (evo IC {evo_ic:+.4f})")
        print("   ", prog.to_string(max_len=120))

        metrics = backtest_cross_sectional_alpha(
            prog=prog,
            aligned_dfs=aligned_dfs,
            common_time_index=common_index,
            stock_symbols=stock_symbols,
            n_stocks=len(stock_symbols),
            fee_bps=cfg.fee,
            lag=cfg.eval_lag,
            hold=cfg.hold,
            scale_method=cfg.scale,
            initial_state_vars_config={"prev_s1_vec": "vector"},
            scalar_feature_names=SCALAR_FEATURE_NAMES,
            cross_sectional_feature_vector_names=CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
            debug_prints=cli.debug_prints,
            annualization_factor=cli.annualization_factor_override
                                   if cli.annualization_factor_override is not None
                                   else cfg.annualization_factor,
        )

        metrics.update({
            "AlphaID":        f"Alpha_{idx:02d}",
            "OriginalMetric": evo_ic,
            "Program":        prog.to_string(max_len=1_000_000_000),
        })
        results.append(metrics)

        print(f"  └─ Sharpe {metrics['Sharpe']:+.3f}  "
              f"AnnRet {metrics['AnnReturn']*100:6.2f}%  "
              f"MaxDD {metrics['MaxDD']*100:6.2f}%  "
              f"Turnover {metrics['Turnover']:.4f}")

    # --------------------------------------------------------- save summary
    if results:
        df = (pd.DataFrame(results)
                .sort_values("Sharpe", ascending=False))
        summary = Path(cli.outdir) / f"backtest_summary_top{cfg.top_to_backtest}.csv"
        df.to_csv(summary, index=False, float_format="%.4f")
        print(f"\nBack-test summary written → {summary}")
        print(df.drop(columns=["Program", "Error"], errors="ignore").to_string(index=False))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
