from __future__ import annotations
import argparse
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
    from evolve_alphas import INITIAL_STATE_VARS as EVO_INITIAL_STATE_VARS, args as evo_args_global
except ImportError:
    EVO_INITIAL_STATE_VARS = {"prev_s1_vec": "vector"} # Minimal fallback
    evo_args_global = None

# Use INITIAL_STATE_VARS from evolve_alphas if available, otherwise a default.
INITIAL_STATE_VARS_BT: Dict[str, str] = EVO_INITIAL_STATE_VARS if EVO_INITIAL_STATE_VARS else {
    "prev_s1_vec": "vector", # A common example
}

DEFAULT_PICKLE_FILE = "evolved_top_alphas.pkl"
DEFAULT_DATA_DIR = "./data"

# Module-level global for parsed arguments of this script
bt_args: argparse.Namespace | None = None 

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


def main() -> None:
    global bt_args 
    bt_ap = argparse.ArgumentParser(description="Back-test evolved cross-sectional alphas")
    bt_ap.add_argument("--input", default=DEFAULT_PICKLE_FILE, help="Pickle file with evolved AlphaPrograms")
    bt_ap.add_argument("--top", type=int, default=10, help="Number of top programs to backtest")
    bt_ap.add_argument("--data", default=DEFAULT_DATA_DIR, help="Directory with *.csv OHLC data")
    bt_ap.add_argument("--fee", type=float, default=1.0, help="Round-trip commission in bps")
    bt_ap.add_argument("--hold", type=int, default=1, help="Holding period in bars")
    bt_ap.add_argument("--scale", choices=["zscore", "rank", "sign"], default="zscore", help="Signal scaling method")
    bt_ap.add_argument("--lag", type=int, default=1, help="Signal lag in bars for position taking")
    bt_ap.add_argument("--outdir", default="evolved_bt_cs_results", help="Directory for output CSVs")
    bt_ap.add_argument("--data_alignment_strategy", type=str, choices=['common_1200', 'specific_long_10k', 'full_overlap'], default='common_1200')
    bt_ap.add_argument("--min_common_data_points", type=int, default=1200)
    bt_ap.add_argument("--seed", type=int, default=None, help="RNG seed for backtest (if programs have stochastic elements)")
    bt_ap.add_argument("--debug_prints", action="store_true", help="Enable debug prints in core logic")
    bt_ap.add_argument("--annualization_factor_override", type=float, default=None, help="Override annualization factor (e.g., 252 for daily, 252*6 for 4H)")

    bt_args = bt_ap.parse_args()

    Path(bt_args.outdir).mkdir(parents=True, exist_ok=True)

    current_backtest_seed = bt_args.seed
    if current_backtest_seed is None and evo_args_global and hasattr(evo_args_global, 'seed'):
        current_backtest_seed = evo_args_global.seed
    
    if current_backtest_seed is not None:
        print(f"Using seed {current_backtest_seed} for backtesting (affects NumPy/random if used by AlphaPrograms).")
        random.seed(current_backtest_seed)
        np.random.seed(current_backtest_seed)


    print(f"Loading and aligning data for backtesting from: {bt_args.data}...")
    aligned_dfs, common_index, stock_symbols = load_and_align_data_for_backtest(
        bt_args.data, bt_args.data_alignment_strategy, bt_args.min_common_data_points
    )
    n_stocks_bt = len(stock_symbols)
    print(f"Backtesting on {n_stocks_bt} symbols over {len(common_index)} time steps.")
    if common_index is not None and len(common_index) > 0:
        print(f"Data spans from {common_index.min()} to {common_index.max()}.")

    if bt_args.debug_prints:
        print("\n--- DEBUG: Verifying ret_fwd (Main Script) ---")
        for sym, df_debug in aligned_dfs.items():
            print(f"Symbol: {sym}, ret_fwd stats:\n{df_debug['ret_fwd'].describe()}")
            if df_debug['ret_fwd'].isnull().any():
                print(f"WARNING: NaNs found in ret_fwd for {sym} after alignment.")
        print("--- END DEBUG: Verifying ret_fwd ---\n")

    programs_to_backtest = load_programs_from_pickle(bt_args.top, bt_args.input)
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
            fee_bps=bt_args.fee,
            lag=bt_args.lag,
            hold=bt_args.hold,
            scale_method=bt_args.scale,
            initial_state_vars_config=INITIAL_STATE_VARS_BT,
            scalar_feature_names=SCALAR_FEATURE_NAMES,
            cross_sectional_feature_vector_names=CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
            debug_prints=bt_args.debug_prints,
            annualization_factor=bt_args.annualization_factor_override if bt_args.annualization_factor_override is not None else (252 * 6) # Default if not overridden
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

        summary_csv_path = Path(bt_args.outdir) / f"backtest_summary_top{bt_args.top}.csv"
        try:
            results_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
            print(f"\nBacktest summary saved to: {summary_csv_path}")
        except Exception as e_csv:
            print(f"Error saving CSV summary: {e_csv}")
            
        print("\n--- Backtest Summary ---")
        print(results_df.drop(columns=["Program", "Error"], errors='ignore').to_string(index=False))

if __name__ == "__main__":
    main()