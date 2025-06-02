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

from config import EvoConfig 
import evolve_alphas as ae
import backtest_evolved_alphas as bt

BASE_OUTPUT_DIR = Path("./pipeline_runs_cs")

def parse_args() -> EvoConfig:
    p = argparse.ArgumentParser(description="Evolve & back-test cross-sectional alphas. Main entry point.")
    
    # Core Evolution arguments
    p.add_argument("generations", type=int, help="# evolutionary generations (e.g., 10)")
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="RNG seed")
    p.add_argument("--data_dir", type=str, default=argparse.SUPPRESS, help="Directory with *.csv OHLC data")
    p.add_argument("--max_lookback_data_option", type=str, 
                    choices=['common_1200', 'specific_long_10k', 'full_overlap'], 
                    default=argparse.SUPPRESS, help="Data alignment strategy")
    p.add_argument("--min_common_points", type=int, default=argparse.SUPPRESS,
                    help="Min common recent data points for data alignment")
    p.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS, help="Hide progress bars")

    # EA parameters
    p.add_argument("--pop_size", type=int, default=argparse.SUPPRESS, help="Population size")
    p.add_argument("--tournament_k", type=int, default=argparse.SUPPRESS, help="Tournament selection K")
    p.add_argument("--p_mut", type=float, default=argparse.SUPPRESS, help="Mutation probability")
    p.add_argument("--p_cross", type=float, default=argparse.SUPPRESS, help="Crossover probability")
    p.add_argument("--elite_keep", type=int, default=argparse.SUPPRESS, help="Number of elites to keep")
    p.add_argument("--fresh_rate", type=float, default=argparse.SUPPRESS, help="Novelty injection rate")
    p.add_argument("--max_ops", type=int, default=argparse.SUPPRESS, help="Max ops per program")
    p.add_argument("--parsimony_penalty", type=float, default=argparse.SUPPRESS, help="Parsimony penalty factor")
    p.add_argument("--corr_penalty_w", type=float, default=argparse.SUPPRESS, help="Correlation penalty weight for HOF")
    p.add_argument("--corr_cutoff", type=float, default=argparse.SUPPRESS, help="Correlation cutoff for HOF penalty")
    p.add_argument("--hof_size", type=int, default=argparse.SUPPRESS, help="Hall of Fame size")
    p.add_argument("--eval_lag", type=int, default=argparse.SUPPRESS, help="Eval IC lag / BT signal lag")
    p.add_argument("--scale", type=str, choices=["zscore","rank","sign"], default=argparse.SUPPRESS, help="Signal scaling method for IC and backtest")
    
    # Evaluation constants for evolve_alphas
    p.add_argument("--xs_flat_guard", type=float, default=argparse.SUPPRESS, help="XS flatness guard threshold")
    p.add_argument("--t_flat_guard", type=float, default=argparse.SUPPRESS, help="Temporal flatness guard threshold")
    p.add_argument("--early_abort_bars", type=int, default=argparse.SUPPRESS, help="Bars for early abort check")
    p.add_argument("--early_abort_xs", type=float, default=argparse.SUPPRESS, help="XS std threshold for early abort")
    p.add_argument("--early_abort_t", type=float, default=argparse.SUPPRESS, help="Temporal std threshold for early abort")
    p.add_argument("--keep_dupes_in_hof", 
                    action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                    help="Allow/disallow duplicate programs (by fingerprint) in HOF")

    # Backtesting specific arguments
    p.add_argument("--top_to_backtest", type=int, default=argparse.SUPPRESS, help="# best programs to backtest")
    p.add_argument("--fee", type=float, default=argparse.SUPPRESS, help="Round-trip commission in bps for backtest")
    p.add_argument("--hold", type=int, default=argparse.SUPPRESS, help="Holding period in bars for backtest")

    parsed_cli_args = p.parse_args()
    # EvoConfig will fill in any missing optional arguments with its own defaults
    # because SUPPRESS ensures they are not in vars(parsed_cli_args) if omitted by user.
    return EvoConfig(**vars(parsed_cli_args))

def _evolve_and_save(
    cfg: EvoConfig,
    run_output_dir: Path
) -> Path:    
    print(f"\n--- Starting Evolution ({cfg.generations} gens, seed {cfg.seed}) ---")
    
    top_evolved_programs_with_ic = ae.evolve(cfg)
    
    top_to_save_and_backtest = top_evolved_programs_with_ic[:cfg.hof_size]
    
    pickle_dir = run_output_dir / "pickles"
    pickle_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_filename = f"evolved_top{cfg.hof_size}_{cfg.generations}g_{cfg.max_lookback_data_option}_{stamp}.pkl"
    pickle_filepath = pickle_dir / out_filename
    
    with open(pickle_filepath, "wb") as fh:
        pickle.dump(top_to_save_and_backtest, fh)
    print(f"\nSaved {len(top_to_save_and_backtest)} programs (with ICs) to: {pickle_filepath}\n")
    return pickle_filepath

def main() -> None:
    cfg = parse_args()
    
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = BASE_OUTPUT_DIR / f"run_g{cfg.generations}_seed{cfg.seed}_{cfg.max_lookback_data_option}_{run_timestamp}"
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    
    evolved_pickle_filepath = _evolve_and_save(
        cfg=cfg,
        run_output_dir=current_run_output_dir
    )
    
    backtest_csv_outdir = current_run_output_dir / "backtest_portfolio_csvs"
    # original_sys_argv = sys.argv[:] # Removed: No longer manipulating sys.argv
    
    # --- Prepare EvoConfig for backtesting ---
    # The cfg object from parse_args() should have most attributes needed by backtest_evolved_alphas.
    # We just need to update the specific paths determined dynamically in this script.
    # It's assumed that EvoConfig (from config.py) has attributes like:
    # input_pickle_file, output_dir, top_n_programs, fee_bps, holding_period, 
    # scaling_method, signal_lag, data_alignment_strategy, min_common_data_points, 
    # data_dir, seed, debug_prints, annualization_factor_override.
    # And that parse_args() populates them correctly (e.g. cfg.top_to_backtest is assigned to cfg.top_n_programs by EvoConfig constructor or similar).

    cfg.input_pickle_file = str(evolved_pickle_filepath)
    cfg.output_dir = str(backtest_csv_outdir)
    
    # The following attributes are assumed to be correctly named and populated on cfg 
    # by EvoConfig's initialization using parameters from parse_args(), for example:
    # cfg.top_n_programs = cfg.top_to_backtest (or EvoConfig uses top_to_backtest directly if that's the name)
    # cfg.signal_lag = cfg.eval_lag
    # cfg.holding_period = cfg.hold
    # cfg.fee_bps = cfg.fee
    # cfg.scaling_method = cfg.scale
    # cfg.data_alignment_strategy = cfg.max_lookback_data_option
    # cfg.min_common_data_points = cfg.min_common_points
    # cfg.data_dir, cfg.seed, cfg.debug_prints should also be correctly set.

    # bt_sys_argv list and sys.argv manipulation removed.

    print(f"\n--- Running Backtests ---")
    # sys.argv = bt_sys_argv # Removed
    
    try:
        # Call bt.main directly with the cfg object
        bt.main(cfg)
    except SystemExit as e:
        if e.code != 0 and e.code is not None: 
            print(f"Backtesting exited with code {e.code}")
    # finally: # Removed
        # sys.argv = original_sys_argv # Removed

    print(f"\n--- Pipeline run finished. Check outputs in: {current_run_output_dir} ---")

if __name__ == "__main__":
    main()