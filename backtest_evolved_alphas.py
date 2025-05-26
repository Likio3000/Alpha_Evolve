# --- START FILE: backtest_evolved_alphas.py ---
#!/usr/bin/env python
"""
backtest_evolved_alphas.py  ·  v1.0 (Cross-Sectional Update)
────────────────────────────────────────────────────────────────────────────
Back-test the top-N AlphaProgram objects saved by evolve_alphas.py,
using a cross-sectional portfolio construction strategy.
"""
from __future__ import annotations
import argparse
import pickle
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Any, OrderedDict as OrderedDictType
from collections import OrderedDict
import math
import random

import numpy as np
import pandas as pd

# --- Safe import of evolve_alphas and alpha_program_core ---
try:
    from evolve_alphas import (
        load_and_align_data,
        _ALIGNED_DFS as EvoAlignedDfs,
        _COMMON_TIME_INDEX as EvoCommonTimeIndex,
        _STOCK_SYMBOLS as EvoStockSymbols,
        _N_STOCKS as EvoNStocks,
        FEATURE_VARS,
        INITIAL_STATE_VARS,
        args as evo_args
    )
    from alpha_program_core import AlphaProgram, FINAL_PREDICTION_VECTOR_NAME, CROSS_SECTIONAL_FEATURE_VECTOR_NAMES, SCALAR_FEATURE_NAMES
    ALPHAEVOLVE_LOADED = True
except ImportError as e:
    print(f"Could not import from evolve_alphas or alpha_program_core: {e}")
    print("Ensure these files are in PYTHONPATH. Backtesting might be limited.")
    ALPHAEVOLVE_LOADED = False
    AlphaProgram = None
    SCALAR_FEATURE_NAMES = ["const_1", "const_0", "const_neg_1"] # Fallback, ensure it's consistent
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES = []
    INITIAL_STATE_VARS = {}
    evo_args = None

DEFAULT_PICKLE_FILE = "evolved_top_alphas.pkl"
DEFAULT_DATA_DIR = "./data"

# ──────────────────────────────────────────────────────────────────────────
# Backtesting Core Logic (Cross-Sectional)
# ──────────────────────────────────────────────────────────────────────────

def _scale_signal_cross_sectionally(raw_signal_vector: np.ndarray, method: str) -> np.ndarray:
    if raw_signal_vector.size == 0:
        return raw_signal_vector
    
    # Replace NaN/inf in raw signals before scaling to avoid issues
    # This is important because NaNs can propagate through mean/std in z-score
    # or cause issues in ranking.
    clean_signal_vector = np.nan_to_num(raw_signal_vector, nan=0.0, posinf=0.0, neginf=0.0)

    if method == "sign":
        scaled = np.sign(clean_signal_vector)
    elif method == "rank":
        if clean_signal_vector.size <= 1:
            scaled = np.zeros_like(clean_signal_vector)
        else:
            temp = clean_signal_vector.argsort()
            ranks = np.empty_like(temp, dtype=float)
            ranks[temp] = np.arange(len(clean_signal_vector))
            scaled = (ranks / (len(clean_signal_vector) - 1 + 1e-9)) * 2.0 - 1.0
    else: # Default: z-score
        mu = np.nanmean(clean_signal_vector) # nanmean is robust to NaNs if any slip through
        sd = np.nanstd(clean_signal_vector)
        if sd < 1e-9 :
            scaled = np.zeros_like(clean_signal_vector)
        else:
            scaled = (clean_signal_vector - mu) / sd
    
    return np.clip(scaled, -1, 1)


def _max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0: return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-9)
    return np.min(drawdown) if len(drawdown) > 0 else 0.0


def backtest_cross_sectional_alpha(
    prog: AlphaProgram,
    aligned_dfs: OrderedDictType[str, pd.DataFrame],
    common_time_index: pd.DatetimeIndex,
    stock_symbols: List[str],
    n_stocks: int,
    fee_bps: float,
    lag: int,
    hold: int,
    scale_method: str,
    current_seed: Optional[int] = None
) -> Dict[str, Any]:
    program_state: Dict[str, Any] = prog.new_state()
    for s_name, s_type in INITIAL_STATE_VARS.items():
        if s_name not in program_state:
            if s_type == "vector": program_state[s_name] = np.zeros(n_stocks)
            else: program_state[s_name] = 0.0

    raw_signals_over_time: List[np.ndarray] = []

    if current_seed is not None:
        random.seed(current_seed)

    for t_idx, timestamp in enumerate(common_time_index):
        if t_idx == len(common_time_index) - 1: break

        features_at_t: Dict[str, Any] = {}
        for feat_template in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
            col_name = feat_template.replace('_t', '')
            try:
                feature_vector = np.array([aligned_dfs[sym].loc[timestamp, col_name] for sym in stock_symbols], dtype=float)
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
                features_at_t[feat_template] = feature_vector
            except KeyError:
                features_at_t[feat_template] = np.zeros(n_stocks, dtype=float)
            except Exception:
                features_at_t[feat_template] = np.zeros(n_stocks, dtype=float)

        for scalar_feat_name in SCALAR_FEATURE_NAMES:
            if scalar_feat_name == "const_1": features_at_t[scalar_feat_name] = 1.0
            elif scalar_feat_name == "const_0": features_at_t[scalar_feat_name] = 0.0
            elif scalar_feat_name == "const_neg_1": features_at_t[scalar_feat_name] = -1.0
            # market_sentiment_idx was removed based on previous feedback

        try:
            signal_vector_t = prog.eval(features_at_t, program_state, n_stocks)
            signal_vector_t = np.nan_to_num(signal_vector_t, nan=0.0, posinf=0.0, neginf=0.0) # Clean output
            raw_signals_over_time.append(signal_vector_t)
        except Exception as e:
            raw_signals_over_time.append(np.zeros(n_stocks))
    
    if not raw_signals_over_time:
        return {"Sharpe": 0.0, "AnnReturn": 0.0, "AnnVol": 0.0, "MaxDD": 0.0, "Turnover": 0.0, "Bars": 0, "Error": "No signals generated"}

    signal_matrix = np.array(raw_signals_over_time)

    # CONSULTANT STEP 3 (Partial): Print cross-sectional std-dev of the raw signal
    print(f"Debug: Raw signal_matrix σ_cross_sectional per bar (first 5): {signal_matrix.std(axis=1)[:5]}")


    target_positions_matrix = np.zeros_like(signal_matrix)
    for t in range(signal_matrix.shape[0]):
        # Scale first
        scaled_signal_t = _scale_signal_cross_sectionally(signal_matrix[t, :], scale_method)
        
        # CONSULTANT STEP 2: Dollar-neutral re-centering
        # Ensure sum of weights is zero, and sum of absolute weights is 1 (or close to it)
        # scaled_signal_t is already clipped to [-1, 1] by _scale_signal_cross_sectionally
        
        # Re-center (make sum of weights zero)
        mean_signal_t = np.mean(scaled_signal_t)
        centered_signal_t = scaled_signal_t - mean_signal_t
        
        # Normalize to make sum of absolute weights = 1 (long-short portfolio)
        # This makes it a "fully invested" long-short portfolio.
        # If centered_signal_t is all zeros (e.g. after scaling a constant signal), sum_abs will be 0.
        sum_abs_centered_signal = np.sum(np.abs(centered_signal_t))
        if sum_abs_centered_signal > 1e-9: # Avoid division by zero
            neutralized_signal_t = centered_signal_t / sum_abs_centered_signal
        else: # If all signals were identical initially, after centering they are all zero
            neutralized_signal_t = np.zeros_like(centered_signal_t)
            
        target_positions_matrix[t, :] = neutralized_signal_t
        # target_positions_matrix[t, :] = scaled_signal_t # Original line before consultant's step 2

    # CONSULTANT STEP 3 (Partial): Print cross-sectional std-dev of the *target positions*
    print(f"Debug: Target_positions_matrix σ_cross_sectional per bar (first 5): {target_positions_matrix.std(axis=1)[:5]}")


    if hold > 1:
        df_target_pos = pd.DataFrame(target_positions_matrix)
        df_held_pos = df_target_pos.rolling(window=hold, min_periods=1).mean()
        # Re-neutralize and re-normalize after rolling mean if desired, as rolling can de-neutralize.
        # For now, let's keep it simple and see the effect of initial neutralization.
        target_positions_matrix = df_held_pos.values
    
    actual_positions = np.zeros_like(target_positions_matrix)
    if lag > 0 and target_positions_matrix.shape[0] > lag:
        actual_positions[lag:, :] = target_positions_matrix[:-lag, :]
    elif lag == 0:
        actual_positions = target_positions_matrix
    
    ret_fwd_matrix = np.zeros_like(actual_positions)
    for i, sym in enumerate(stock_symbols):
        # Ensure ret_fwd is clean of NaNs/Infs from data source or calculation
        ret_fwd_values = aligned_dfs[sym]["ret_fwd"].loc[common_time_index[:signal_matrix.shape[0]]].values
        ret_fwd_matrix[:, i] = np.nan_to_num(ret_fwd_values, nan=0.0, posinf=0.0, neginf=0.0)

    daily_portfolio_returns = np.sum(actual_positions * ret_fwd_matrix, axis=1)
    
    pos_diff = np.diff(actual_positions, axis=0, prepend=np.zeros((1, n_stocks)))
    abs_pos_diff_sum = np.sum(np.abs(pos_diff), axis=1)
    
    transaction_costs = (fee_bps * 1e-4) * abs_pos_diff_sum
    daily_portfolio_returns_net = daily_portfolio_returns - transaction_costs

    # CONSULTANT STEP 1: Print daily PnL and equity curve details
    if len(daily_portfolio_returns_net) > 0:
        mean_ret_calc = np.mean(daily_portfolio_returns_net)
        std_ret_calc = np.std(daily_portfolio_returns_net, ddof=0) # Use ddof=0 for population std
        print(f"DEBUG: first 20 daily PnL   : {daily_portfolio_returns_net[:20]}")
        equity_curve_debug = np.cumprod(1 + daily_portfolio_returns_net) # Calculate for debug
        print(f"DEBUG: first 20 eq. curve   : {equity_curve_debug[:20]}")
        print(f"DEBUG: mean {mean_ret_calc:.6e}  std {std_ret_calc:.6e}")
    # End CONSULTANT STEP 1

    if len(daily_portfolio_returns_net) < 2:
        return {"Sharpe": 0.0, "AnnReturn": 0.0, "AnnVol": 0.0, "MaxDD": 0.0, "Turnover": 0.0, "Bars": len(daily_portfolio_returns_net)}

    equity_curve = np.cumprod(1 + daily_portfolio_returns_net) # Recalculate equity curve for metrics
    
    mean_ret = np.mean(daily_portfolio_returns_net)
    std_ret = np.std(daily_portfolio_returns_net, ddof=0)

    annualization_factor = 365 * (24/4)
    
    sharpe_ratio = (mean_ret / (std_ret + 1e-9)) * np.sqrt(annualization_factor)
    
    total_return = equity_curve[-1] - 1
    num_years = len(daily_portfolio_returns_net) / annualization_factor
    annualized_return = ((1 + total_return) ** (1 / num_years)) - 1 if num_years > 0 else 0.0
    
    annualized_volatility = std_ret * np.sqrt(annualization_factor)
    max_dd = _max_drawdown(equity_curve)
    
    avg_daily_turnover = np.mean(abs_pos_diff_sum) / (n_stocks + 1e-9)

    return {
        "Sharpe": sharpe_ratio,
        "AnnReturn": annualized_return,
        "AnnVol": annualized_volatility,
        "MaxDD": max_dd,
        "Turnover": avg_daily_turnover,
        "Bars": len(daily_portfolio_returns_net)
    }

# ──────────────────────────────────────────────────────────────────────────
def load_programs_from_pickle(n_to_load: int, pickle_filepath: str) \
        -> List[Tuple[AlphaProgram, float]]:
    if not Path(pickle_filepath).exists():
        sys.exit(f"Pickle file not found: {pickle_filepath}")
    with open(pickle_filepath, "rb") as fh:
        loaded_data: List[Tuple[AlphaProgram, float]] = pickle.load(fh)
    return loaded_data[:n_to_load]

# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    bt_ap = argparse.ArgumentParser(description="Back-test evolved cross-sectional alphas")
    bt_ap.add_argument("--input", default=DEFAULT_PICKLE_FILE, help="Pickle file with evolved AlphaPrograms")
    bt_ap.add_argument("--top", type=int, default=10, help="Number of top programs to backtest from the pickle file")
    bt_ap.add_argument("--data", default=DEFAULT_DATA_DIR, help="Directory with *.csv OHLC data")
    bt_ap.add_argument("--fee", type=float, default=1.0, help="Round-trip commission in bps (e.g., 1.0 for 0.01%)")
    bt_ap.add_argument("--hold", type=int, default=1, help="Holding period in bars (1 = one-bar hold)")
    bt_ap.add_argument("--scale", choices=["zscore", "rank", "sign"], default="zscore", help="Signal scaling method")
    bt_ap.add_argument("--lag", type=int, default=1, help="Signal lag in bars for position taking")
    bt_ap.add_argument("--outdir", default="evolved_bt_cs_results", help="Directory for output CSVs")
    bt_ap.add_argument("--data_alignment_strategy", type=str, choices=['common_1200', 'specific_long_10k', 'full_overlap'], default='common_1200')
    bt_ap.add_argument("--min_common_data_points", type=int, default=1200)
    bt_ap.add_argument("--seed", type=int, default=None, help="RNG seed for features (optional, for reproducibility)")
    bt_ap.add_argument("--debug_prints", action="store_true", help="Enable consultant's debug prints")


    bt_args = bt_ap.parse_args()

    if not ALPHAEVOLVE_LOADED and AlphaProgram is None:
        sys.exit("AlphaProgram class not available. Cannot proceed with backtesting.")

    Path(bt_args.outdir).mkdir(parents=True, exist_ok=True)

    current_backtest_seed = bt_args.seed
    if current_backtest_seed is None and evo_args and hasattr(evo_args, 'seed'):
        current_backtest_seed = evo_args.seed

    print(f"Loading and aligning data for backtesting from: {bt_args.data}...")
    aligned_dfs, common_index, stock_symbols = load_and_align_data(
        bt_args.data, bt_args.data_alignment_strategy, bt_args.min_common_data_points
    )
    n_stocks_bt = len(stock_symbols)
    print(f"Backtesting on {n_stocks_bt} symbols over {len(common_index)} time steps.")

    # CONSULTANT STEP 4: Verify ret_fwd
    if bt_args.debug_prints:
        print("\n--- DEBUG: Verifying ret_fwd (Consultant Step 4) ---")
        for sym, df_debug in aligned_dfs.items():
            print(f"Symbol: {sym}, ret_fwd stats:\n{df_debug['ret_fwd'].describe()}")
        print("--- END DEBUG: Verifying ret_fwd ---\n")


    programs_to_backtest = load_programs_from_pickle(bt_args.top, bt_args.input)

    all_results = []
    for idx, (prog, original_metric) in enumerate(programs_to_backtest, 1):
        print(f"\nBacktesting Alpha #{idx:02d} (Original Metric: {original_metric:+.4f})")
        print(f"   {textwrap.shorten(prog.to_string(), 1000)}")

        metrics = backtest_cross_sectional_alpha(
            prog,
            aligned_dfs,
            common_index,
            stock_symbols,
            n_stocks_bt,
            fee_bps=bt_args.fee,
            lag=bt_args.lag,
            hold=bt_args.hold,
            scale_method=bt_args.scale,
            current_seed=current_backtest_seed
        )
        
        metrics["AlphaID"] = f"Alpha_{idx:02d}"
        metrics["OriginalMetric"] = original_metric
        metrics["Program"] = prog.to_string(max_len=1000)
        all_results.append(metrics)

        print(f"  └─ Sharpe: {metrics.get('Sharpe', 0.0):.3f}, AnnReturn: {metrics.get('AnnReturn', 0.0)*100:.2f}%, "
              f"MaxDD: {metrics.get('MaxDD', 0.0)*100:.2f}%, Turnover: {metrics.get('Turnover', 0.0):.4f}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        cols_order = ["AlphaID", "Sharpe", "AnnReturn", "AnnVol", "MaxDD", "Turnover", "Bars", "OriginalMetric", "Program", "Error"]
        cols_present = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[cols_present]
        results_df = results_df.sort_values("Sharpe", ascending=False)

        summary_csv_path = Path(bt_args.outdir) / f"backtest_summary_top{bt_args.top}.csv"
        results_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
        print(f"\n 종합 Backtest summary saved to: {summary_csv_path}")
        print(results_df.drop(columns=["Program", "Error"], errors='ignore').to_string(index=False))

if __name__ == "__main__":
    main()
