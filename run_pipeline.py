#!/usr/bin/env python
"""
run_pipeline.py – Evolve cross-sectional alphas *and* immediately back-test them.
Usage:  uv run run_pipeline.py 5 --max_lookback_data_option common_1200 --fee 1.0 --top 10
"""
from __future__ import annotations
import argparse
import pickle
import sys
import time
from pathlib import Path

current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import evolve_alphas as ae
import backtest_evolved_alphas as bt

BASE_OUTPUT_DIR = Path("./pipeline_runs_cs")

def _evolve_and_save(
    pipeline_args: argparse.Namespace, # Pass the full pipeline args object
    run_output_dir: Path
) -> Path:    
    # `ae.args` is already set to `pipeline_args` in `main()` before calling this function.
    # `evolve_alphas.py` will use `ae.args` to configure itself internally,
    # especially its constants, via `_sync_constants_from_args()` called by `_ensure_data_loaded()`.

    print(f"\n--- Starting Evolution ({pipeline_args.generations} gens, seed {pipeline_args.seed}) ---")
    
    top_evolved_programs_with_ic = ae.evolve()
    
    top_to_save_and_backtest = top_evolved_programs_with_ic[:pipeline_args.hof_size] # Use hof_size from pipeline_args
    
    pickle_dir = run_output_dir / "pickles"
    pickle_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    # Use pipeline_args for filename consistency
    out_filename = f"evolved_top{pipeline_args.hof_size}_{pipeline_args.generations}g_{pipeline_args.max_lookback_data_option}_{stamp}.pkl"
    pickle_filepath = pickle_dir / out_filename
    
    with open(pickle_filepath, "wb") as fh:
        pickle.dump(top_to_save_and_backtest, fh)
    print(f"\nSaved {len(top_to_save_and_backtest)} programs (with ICs) to: {pickle_filepath}\n")
    return pickle_filepath

def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve & back-test cross-sectional alphas in one go")
    
    # Evolution specific or shared that need to be passed to evolve_alphas
    # These names must match those in evolve_alphas._parse_cli() if ae.args is to be a drop-in replacement.
    ap.add_argument("generations", type=int, help="# evolutionary generations")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for evolution")
    
    # Get default from evolve_alphas.args if it has been initialized, otherwise a sensible default.
    default_data_dir = getattr(ae.args, 'data_dir', "./data") if hasattr(ae, 'args') else "./data"
    ap.add_argument("--data_dir", default=default_data_dir, help="Directory with *.csv OHLC data")
    
    ap.add_argument("--max_lookback_data_option", type=str, choices=['common_1200', 'specific_long_10k', 'full_overlap'], 
                    default=getattr(ae.args, 'max_lookback_data_option', 'common_1200') if hasattr(ae, 'args') else 'common_1200', 
                    help="Data alignment strategy for evolution and backtesting")
    ap.add_argument("--min_common_points", type=int, # Renamed from min_common_data_points
                    default=getattr(ae.args, 'min_common_points', 1200) if hasattr(ae, 'args') else 1200, 
                    help="Min common recent data points for data alignment")
    ap.add_argument("--quiet", action="store_true", default=getattr(ae.args, 'quiet', False) if hasattr(ae, 'args') else False, help="Hide progress bars")

    # EA parameters
    ap.add_argument("--pop_size", type=int, default=getattr(ae.args, 'pop_size', 64) if hasattr(ae, 'args') else 64)
    ap.add_argument("--tournament_k", type=int, default=getattr(ae.args, 'tournament_k', 5) if hasattr(ae, 'args') else 5)
    ap.add_argument("--p_mut", type=float, default=getattr(ae.args, 'p_mut', 0.4) if hasattr(ae, 'args') else 0.4)
    ap.add_argument("--p_cross", type=float, default=getattr(ae.args, 'p_cross', 0.6) if hasattr(ae, 'args') else 0.6)
    ap.add_argument("--elite_keep", type=int, default=getattr(ae.args, 'elite_keep', 4) if hasattr(ae, 'args') else 4)
    ap.add_argument("--max_ops", type=int, default=getattr(ae.args, 'max_ops', 32) if hasattr(ae, 'args') else 32)
    ap.add_argument("--parsimony_penalty", type=float, default=getattr(ae.args, 'parsimony_penalty', 0.01) if hasattr(ae, 'args') else 0.01)
    ap.add_argument("--corr_penalty_w", type=float, default=getattr(ae.args, 'corr_penalty_w', 0.15) if hasattr(ae, 'args') else 0.15)
    ap.add_argument("--corr_cutoff", type=float, default=getattr(ae.args, 'corr_cutoff', 0.20) if hasattr(ae, 'args') else 0.20)
    ap.add_argument("--hof_size", type=int, default=getattr(ae.args, 'hof_size', 20) if hasattr(ae, 'args') else 20, help="Size of Hall of Fame / Num top programs to save for backtest.")
    ap.add_argument("--eval_lag", type=int, default=getattr(ae.args, 'eval_lag', 1) if hasattr(ae, 'args') else 1, help="Lag for IC calculation during evolution.")

    # Backtesting specific arguments
    ap.add_argument("--top_to_backtest", type=int, default=10, help="# best programs from evolution to backtest")
    ap.add_argument("--fee", type=float, default=1.0, help="Round-trip commission in bps for backtest (e.g. 1.0 for 0.01%)")
    ap.add_argument("--hold", type=int, default=1, help="Holding period in bars for backtest")
    ap.add_argument("--scale", default="zscore", choices=["zscore","rank","sign"], help="Signal scaling for backtest")
    # --lag for backtester will be derived from pipeline_args.eval_lag
    
    pipeline_args = ap.parse_args() # Renamed to avoid confusion with ae.args
    
    # Set the `args` object in `evolve_alphas` module.
    # This is the primary way to configure `evolve_alphas.py` when it's imported.
    ae.args = pipeline_args 

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = BASE_OUTPUT_DIR / f"run_g{pipeline_args.generations}_seed{pipeline_args.seed}_{pipeline_args.max_lookback_data_option}_{run_timestamp}"
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    
    evolved_pickle_filepath = _evolve_and_save(
        pipeline_args=pipeline_args,
        run_output_dir=current_run_output_dir
    )
    
    backtest_csv_outdir = current_run_output_dir / "backtest_portfolio_csvs"
    original_sys_argv = sys.argv[:]
    
    bt_sys_argv = [
        "backtest_evolved_alphas.py", # Script name
        "--input", str(evolved_pickle_filepath),
        "--top", str(pipeline_args.top_to_backtest),
        "--fee", str(pipeline_args.fee),
        "--hold", str(pipeline_args.hold),
        "--scale", pipeline_args.scale,
        "--lag", str(pipeline_args.eval_lag), # Fix #3: Pass eval_lag from pipeline_args as --lag to backtester
        "--data", str(pipeline_args.data_dir),
        "--outdir", str(backtest_csv_outdir),
        # Pass the correct attribute from pipeline_args for the backtester's expected arg name
        "--data_alignment_strategy", pipeline_args.max_lookback_data_option, 
        "--min_common_data_points", str(pipeline_args.min_common_points)
    ]
    
    print(f"\n--- Running Backtests ---")
    sys.argv = bt_sys_argv
    
    try:
        bt.main()
    except SystemExit as e:
        print(f"Backtesting exited with code {e.code}")
    finally:
        sys.argv = original_sys_argv

    print(f"\n--- Pipeline run finished. Check outputs in: {current_run_output_dir} ---")

if __name__ == "__main__":
    main()

# To run use:    uv run run_pipeline.py 10 --seed 123 --max_lookback_data_option full_overlap --min_common_points 1200 --pop_size 32 --top_to_backtest 5 --fee 0