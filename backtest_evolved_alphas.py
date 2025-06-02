from __future__ import annotations
# import argparse # Removed
import pickle
import sys
import textwrap # For printing program strings
from pathlib import Path
from typing import Dict, List, Tuple, Any # OrderedDict removed as it's handled in components
import random # For setting seed if specified
import numpy as np # For general numpy usage if any direct manipulation needed (rare now)
import pandas as pd # For DataFrame creation for results

# --- Framework and Component Imports ---
try:
    # From the main alpha generation framework
    from alpha_framework import AlphaProgram, CROSS_SECTIONAL_FEATURE_VECTOR_NAMES, SCALAR_FEATURE_NAMES
    ALPHAFRAMEWORK_LOADED = True
except ImportError as e_af:
    print(f"Critical Error: Could not import from alpha_framework: {e_af}")
    print("Ensure alpha_framework package is in PYTHONPATH. Backtesting cannot proceed.")
    sys.exit(1) # Exit if core framework is missing

try:
    # From our new backtesting components
    from backtesting_components import load_and_align_data_for_backtest, backtest_cross_sectional_alpha
    BACKTESTING_COMPONENTS_LOADED = True
except ImportError as e_bc:
    print(f"Critical Error: Could not import from backtesting_components: {e_bc}")
    print("Ensure backtesting_components package is in PYTHONPATH. Backtesting cannot proceed.")
    sys.exit(1) # Exit if backtesting components are missing

# Attempt to import INITIAL_STATE_VARS and evo_args from evolve_alphas (optional, for context)
try:
    from evolve_alphas import INITIAL_STATE_VARS as EVO_INITIAL_STATE_VARS #, args as evo_args_global # evo_args_global removed
except ImportError:
    EVO_INITIAL_STATE_VARS = {"prev_s1_vec": "vector"} # Minimal fallback
    # evo_args_global = None # evo_args_global removed

# Removed placeholder EvoConfig, will import from config.py
from config import EvoConfig

# Use INITIAL_STATE_VARS from evolve_alphas if available, otherwise a default.
INITIAL_STATE_VARS_BT: Dict[str, str] = EVO_INITIAL_STATE_VARS if EVO_INITIAL_STATE_VARS else {
    "prev_s1_vec": "vector", # A common example
}

DEFAULT_PICKLE_FILE = "evolved_top_alphas.pkl" # Kept as is
DEFAULT_DATA_DIR = "./data" # Kept as is

# Module-level global for parsed arguments of this script - REMOVED
# bt_args: argparse.Namespace | None = None 

def load_programs_from_pickle(n_to_load: int, pickle_filepath: str) \
        -> List[Tuple[AlphaProgram, float]]:
    if not Path(pickle_filepath).exists():
        sys.exit(f"Pickle file not found: {pickle_filepath}")
    try:
        with open(pickle_filepath, "rb") as fh:
            # The loaded data is List[Tuple[AlphaProgram, float (IC from evolution)]]
            loaded_data: List[Tuple[AlphaProgram, float]] = pickle.load(fh)
        return loaded_data[:n_to_load]
    except (pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError) as e:
        sys.exit(f"Error loading or interpreting pickle file {pickle_filepath}: {e}\n"
                 "Ensure the pickle file was created with compatible AlphaProgram versions.")


def main(cfg: EvoConfig) -> None: # Modified signature
    # global bt_args # Removed
    # Argument parsing logic removed, configuration now comes from cfg

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Seed logic now relies on cfg.seed. 
    # The caller of main (e.g. a script that instantiates EvoConfig) would be responsible
    # for potentially incorporating evo_args_global.seed into cfg.seed if that's desired.
    # current_backtest_seed = cfg.seed # Now directly use cfg.seed from the imported EvoConfig
    
    if cfg.seed is not None:
        print(f"Using seed {cfg.seed} for backtesting (affects NumPy/random if used by AlphaPrograms).")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    # cfg.data_dir should be correctly populated by EvoConfig
    print(f"Loading and aligning data for backtesting from: {cfg.data_dir}...") 
    aligned_dfs, common_index, stock_symbols = load_and_align_data_for_backtest(
        cfg.data_dir, cfg.max_lookback_data_option, cfg.min_common_points # Changed to EvoConfig field names
    )
    n_stocks_bt = len(stock_symbols)
    print(f"Backtesting on {n_stocks_bt} symbols over {len(common_index)} time steps.")
    if common_index is not None and len(common_index) > 0:
        print(f"Data spans from {common_index.min()} to {common_index.max()}.")

    if cfg.debug_prints:
        print("\n--- DEBUG: Verifying ret_fwd (Main Script) ---")
        for sym, df_debug in aligned_dfs.items():
            print(f"Symbol: {sym}, ret_fwd stats:\n{df_debug['ret_fwd'].describe()}")
            if df_debug['ret_fwd'].isnull().any():
                print(f"WARNING: NaNs found in ret_fwd for {sym} after alignment.")
        print("--- END DEBUG: Verifying ret_fwd ---\n")

    # Use cfg.top_to_backtest and cfg.input_pickle_file from the imported EvoConfig
    programs_to_backtest = load_programs_from_pickle(cfg.top_to_backtest, cfg.input_pickle_file)
    all_results = []
    
    for idx, (prog, original_metric) in enumerate(programs_to_backtest, 1):
        prog_str_summary = prog.to_string(max_len=100)
        print(f"\nBacktesting Alpha #{idx:02d} (Original Metric from Evo: {original_metric:+.4f})")
        print(f"   Program: {prog_str_summary}")

        metrics = backtest_cross_sectional_alpha(
            prog=prog,
            aligned_dfs=aligned_dfs,
            common_time_index=common_index,
            stock_symbols=stock_symbols,
            n_stocks=n_stocks_bt,
            fee_bps=cfg.fee, # Changed to EvoConfig field name
            lag=cfg.eval_lag, # Changed to EvoConfig field name
            hold=cfg.hold, # Changed to EvoConfig field name
            scale_method=cfg.scale, # Changed to EvoConfig field name
            initial_state_vars_config=INITIAL_STATE_VARS_BT, # Remains as is
            scalar_feature_names=SCALAR_FEATURE_NAMES,
            cross_sectional_feature_vector_names=CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
            debug_prints=cfg.debug_prints, # Directly use from EvoConfig
            annualization_factor=cfg.annualization_factor_override if cfg.annualization_factor_override is not None else (252 * 6) # Directly use from EvoConfig
        )
        
        metrics["AlphaID"] = f"Alpha_{idx:02d}"
        metrics["OriginalMetric"] = original_metric # IC from evolution
        metrics["Program"] = prog.to_string(max_len=1_000_000_000) # Full program string for CSV
        all_results.append(metrics)

        print(f"  └─ Sharpe: {metrics.get('Sharpe', 0.0):.3f}, AnnReturn: {metrics.get('AnnReturn', 0.0)*100:.2f}%, "
              f"MaxDD: {metrics.get('MaxDD', 0.0)*100:.2f}%, Turnover: {metrics.get('Turnover', 0.0):.4f}, Bars: {metrics.get('Bars',0)}")
        if "Error" in metrics and metrics["Error"]:
            print(f"     ERROR: {metrics['Error']}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        cols_order = ["AlphaID", "Sharpe", "AnnReturn", "AnnVol", "MaxDD", "Turnover", "Bars", "OriginalMetric", "Program", "Error"]
        cols_present = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[cols_present]
        
        results_df = results_df.sort_values("Sharpe", ascending=False)

        # Use cfg.output_dir and cfg.top_to_backtest from the imported EvoConfig
        summary_csv_path = Path(cfg.output_dir) / f"backtest_summary_top{cfg.top_to_backtest}.csv"
        try:
            results_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
            print(f"\nBacktest summary saved to: {summary_csv_path}")
        except Exception as e_csv:
            print(f"Error saving CSV summary: {e_csv}")
            
        print("\n--- Backtest Summary ---")
        print(results_df.drop(columns=["Program", "Error"], errors='ignore').to_string(index=False))

# Removed the if __name__ == "__main__": block as this script is not intended to be run directly in the pipeline.
# The main function is called by run_pipeline.py with a fully populated EvoConfig object.