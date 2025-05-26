from __future__ import annotations

"""evolve_alphas.py  ·  v5.3 – Cross-Sectional Evolution with Tighter Guards
======================================================================
Evolutionary search for weakly‑correlated alphas **à la AlphaEvolve** on a
folder of 4‑hour OHLC crypto CSVs, now with cross-sectional evaluation.

Changes vs v5.2 (Consultant-driven May 2024) ──────────────────────────
* **Tighter Flat Signal Penalty**: The threshold for the "flat-over-time"
  signal guard in `evaluate()` has been increased from `1e-5` to `1e-2`.
* **Removed `const_0`**: `const_0` is no longer part of the default scalar
  features available to the AlphaProgram (change made in alpha_program_core.py).
* **A1 (Lag Alignment)**: Added `--eval_lag` CLI arg. `evaluate()` uses this lag
  for fetching forward returns.
* **A2 (Cross-Sectional Flatness Guard)**: Penalizes programs if mean
  cross-sectional signal standard deviation is too low.
* **A3 (Reduce Constant Scalar Weight)**: Probability of `const_1`/`const_neg_1`
  reduced in program generation/mutation (implemented in alpha_program_core.py).
* **B1 (Early Abort)**: Checks for flat signals (cross-sectionally or temporally)
  after `EARLY_ABORT_BARS` and penalizes/aborts.
* **C1 (Tqdm Visibility)**: Progress bar default changed to visible.
* **C2 (ETA in Log)**: Per-generation log includes ETA.
"""

from pathlib import Path
import argparse
import os
import glob
import math
import random
import sys
import textwrap
import time
from typing import Dict, List, Tuple, Optional, Any, OrderedDict as OrderedDictType, Set
from collections import OrderedDict, deque


import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from alpha_program_core import (
    AlphaProgram, Op, TypeId,
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES, 
    SCALAR_FEATURE_NAMES, # This will now be ['const_1', 'const_neg_1']
    FINAL_PREDICTION_VECTOR_NAME,
    OP_REGISTRY 
)
import pickle

###############################################################################
# CLI & CONFIG ################################################################
###############################################################################

def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cross-Sectional Evolutionary Alpha Search")
    ap.add_argument("generations", nargs="?", type=int, default=10)
    ap.add_argument("seed", nargs="?", type=int, default=42)
    ap.add_argument("-q", "--quiet", action="store_true", help="hide progress bar")
    ap.add_argument("--max_lookback_data_option", type=str, choices=['common_1200', 'specific_long_10k', 'full_overlap'], default='common_1200',
                    help="Data alignment strategy: 'common_1200' for min 1200 recent points, "
                         "'specific_long_10k' for files with >10k points, 'full_overlap' for max common history.")
    ap.add_argument("--min_common_points", type=int, default=1200,
                    help="Minimum number of common recent data points for 'common_1200' or 'specific_long_10k'.")
    ap.add_argument("--data_dir", default="./data", help="Directory with *.csv OHLC data")
    ap.add_argument("--pop_size", type=int, default=64)
    ap.add_argument("--tournament_k", type=int, default=5)
    ap.add_argument("--p_mut", type=float, default=0.4)
    ap.add_argument("--p_cross", type=float, default=0.6)
    ap.add_argument("--elite_keep", type=int, default=4)
    ap.add_argument("--max_ops", type=int, default=32)
    ap.add_argument("--parsimony_penalty", type=float, default=0.01)
    ap.add_argument("--corr_penalty_w", type=float, default=0.15)
    ap.add_argument("--corr_cutoff", type=float, default=0.20)
    ap.add_argument("--hof_size", type=int, default=20)
    ap.add_argument("--scale", default="zscore", choices=["zscore","rank","sign"], 
                    help="Signal scaling method for IC calculation (should match backtest)")
    ap.add_argument("--eval_lag", type=int, default=1, 
                    help="Lag (in bars) for fetching forward returns during IC calculation in evaluation. Default 1.")
    return ap.parse_args()

if __name__ == "__main__" or "pytest" in sys.modules:
    args = _parse_cli()
else: 
    class _DefaultArgs:
        # C1: Default quiet to False
        generations = 5; seed = 42; quiet = False
        max_lookback_data_option = 'common_1200'; min_common_points = 1200
        data_dir = "./data"; pop_size = 32; tournament_k = 3
        p_mut = 0.4; p_cross = 0.6; elite_keep = 2; max_ops = 20
        parsimony_penalty = 0.01; corr_penalty_w = 0.15
        corr_cutoff = 0.20; hof_size = 10; scale = "zscore" 
        eval_lag = 1 # Ensure eval_lag exists for _DefaultArgs (Fix #6)
    args = _DefaultArgs()

DATA_DIR = args.data_dir
POP_SIZE = args.pop_size
N_GENERATIONS = args.generations
TOURNAMENT_K = args.tournament_k
P_MUT = args.p_mut
P_CROSS = args.p_cross
ELITE_KEEP = args.elite_keep
MAX_OPS = args.max_ops
PARSIMONY_PENALTY = args.parsimony_penalty
CORR_PENALTY_W = args.corr_penalty_w
CORR_CUTOFF = args.corr_cutoff
DUPLICATE_HOF_SZ = args.hof_size
SEED = args.seed
EVAL_LAG = args.eval_lag # A1

# A2 & B1 constants
XS_FLATNESS_GUARD_THRESHOLD = 1e-2
EARLY_ABORT_BARS = 20
EARLY_ABORT_XS_THRESHOLD = 1e-2 
EARLY_ABORT_T_THRESHOLD = 1e-2 # Can be same as existing flat_signal_threshold
# Consultant suggestion: Configurable HOF uniqueness. Default to False (ensure unique).
# This could be promoted to a CLI argument later if needed.
KEEP_DUPES_IN_HOF_CONFIG = False

random.seed(SEED) 
np.random.seed(SEED) 

###############################################################################
# 1. DATA LOADING & PREPARATION ###############################################
###############################################################################
# FEATURE_VARS reflects the constants available from alpha_program_core.py
FEATURE_VARS: Dict[str, TypeId] = {name: "vector" for name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES}
FEATURE_VARS.update({name: "scalar" for name in SCALAR_FEATURE_NAMES}) # SCALAR_FEATURE_NAMES no longer includes 'const_0'
# Ensure const_1 and const_neg_1 are present if SCALAR_FEATURE_NAMES was modified elsewhere
if "const_1" not in FEATURE_VARS: FEATURE_VARS["const_1"] = "scalar"
if "const_neg_1" not in FEATURE_VARS: FEATURE_VARS["const_neg_1"] = "scalar"


INITIAL_STATE_VARS: Dict[str, TypeId] = {
    "prev_s1_vec": "vector",
    "rolling_mean_custom": "vector"
}

_ALIGNED_DFS: Optional[OrderedDictType[str, pd.DataFrame]] = None
_COMMON_TIME_INDEX: Optional[pd.DatetimeIndex] = None
_STOCK_SYMBOLS: Optional[List[str]] = None
_N_STOCKS: Optional[int] = None

def _sync_constants_from_args():
    global DATA_DIR, POP_SIZE, N_GENERATIONS, TOURNAMENT_K, P_MUT, P_CROSS, ELITE_KEEP
    global MAX_OPS, PARSIMONY_PENALTY, CORR_PENALTY_W, CORR_CUTOFF, DUPLICATE_HOF_SZ, SEED, EVAL_LAG
    # Note: KEEP_DUPES_IN_HOF_CONFIG and other new constants are not currently synced from args, they're global toggles or derived.

    DATA_DIR = args.data_dir
    POP_SIZE = args.pop_size
    N_GENERATIONS = args.generations
    TOURNAMENT_K = args.tournament_k
    P_MUT = args.p_mut
    P_CROSS = args.p_cross
    ELITE_KEEP = args.elite_keep
    MAX_OPS = args.max_ops
    PARSIMONY_PENALTY = args.parsimony_penalty
    CORR_PENALTY_W = args.corr_penalty_w
    CORR_CUTOFF = args.corr_cutoff
    DUPLICATE_HOF_SZ = args.hof_size
    EVAL_LAG = args.eval_lag # A1
    if SEED != args.seed: 
        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)


def _ensure_data_loaded():
    global _ALIGNED_DFS, _COMMON_TIME_INDEX, _STOCK_SYMBOLS, _N_STOCKS, args

    if _ALIGNED_DFS is not None:
        return

    _sync_constants_from_args() 

    df_data, time_idx_data, symbols_data = load_and_align_data(
        args.data_dir, args.max_lookback_data_option, args.min_common_points
    )
    _ALIGNED_DFS = df_data
    _COMMON_TIME_INDEX = time_idx_data
    _STOCK_SYMBOLS = symbols_data
    _N_STOCKS = len(_STOCK_SYMBOLS)

    print(f"evolve_alphas: Using {_N_STOCKS} symbols for evolution: {', '.join(_STOCK_SYMBOLS)}")
    print(f"Data spans {_COMMON_TIME_INDEX.min()} to {_COMMON_TIME_INDEX.max()} with {_COMMON_TIME_INDEX.size} steps. Eval Lag: {EVAL_LAG}")

def _rolling_features_individual_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in (5, 10, 20, 30):
        df[f"ma{w}"] = df["close"].rolling(w, min_periods=1).mean()
        df[f"vol{w}"] = df["close"].rolling(w, min_periods=1).std(ddof=0)
    df["range"] = df["high"] - df["low"]
    # Calculate ret_fwd based on the actual next available close, not shifted by a fixed number
    # The shift for EVAL_LAG happens during its usage in evaluate(), not here.
    df["ret_fwd"] = df["close"].pct_change(periods=1).shift(-1) 
    return df

def load_and_align_data(data_dir_param: str, strategy_param: str, min_common_points_param: int) -> Tuple[OrderedDictType[str, pd.DataFrame], pd.DatetimeIndex, List[str]]:
    raw_dfs: Dict[str, pd.DataFrame] = {}
    for csv_file in glob.glob(os.path.join(data_dir_param, "*.csv")):
        try:
            df = pd.read_csv(csv_file)
            if 'time' not in df.columns:
                continue
            df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
            df = df.dropna(subset=['time']).sort_values("time").set_index("time")
            if df.empty: continue
            df_with_features = _rolling_features_individual_df(df)
            # For ret_fwd, we need to ensure it doesn't have NaNs due to shift at the very end.
            # Dropna here will handle it if features + ret_fwd cause NaNs.
            raw_dfs[Path(csv_file).stem] = df_with_features.dropna() 
        except Exception:
            continue
    
    if not raw_dfs:
        sys.exit(f"No valid CSV data loaded from {data_dir_param}.")

    if strategy_param == 'specific_long_10k':
        min_len_for_long = min_common_points_param 
        raw_dfs = {sym: df for sym, df in raw_dfs.items() if len(df) >= min_len_for_long}
        if len(raw_dfs) < 2:
             sys.exit(f"Not enough long files (>= {min_len_for_long} data points) found for 'specific_long_10k' strategy. Found: {len(raw_dfs)}")

    common_index: Optional[pd.DatetimeIndex] = None
    for df_sym in raw_dfs.values(): 
        if common_index is None: common_index = df_sym.index
        else: common_index = common_index.intersection(df_sym.index)

    # Ensure enough data points *considering the EVAL_LAG for forward returns*
    # If EVAL_LAG is 1, we need at least `min_common_points_param + 1` total points in common_index
    # to have `min_common_points_param` evaluation steps.
    required_length_for_eval = min_common_points_param + EVAL_LAG 
    if common_index is None or len(common_index) < required_length_for_eval: 
        sys.exit(f"Not enough common history (need {required_length_for_eval} for {min_common_points_param} eval steps + lag {EVAL_LAG}, got {len(common_index if common_index is not None else [])}).")

    if strategy_param == 'common_1200' or strategy_param == 'specific_long_10k': 
        # Slice to get `min_common_points_param` *plus* the `EVAL_LAG` lookahead for returns
        num_points_to_keep = min_common_points_param + EVAL_LAG
        if len(common_index) > num_points_to_keep:
            common_index = common_index[-num_points_to_keep:] 
    
    aligned_dfs_ordered = OrderedDict()
    for sym in sorted(raw_dfs.keys()): 
        df_sym = raw_dfs[sym]
        reindexed_df = df_sym.reindex(common_index).ffill().bfill()
        if reindexed_df.isnull().values.any():
             print(f"Warning: DataFrame for {sym} still contains NaNs after ffill/bfill on common_index. This might affect results.")
        aligned_dfs_ordered[sym] = reindexed_df
    
    stock_symbols = list(aligned_dfs_ordered.keys())
    if len(stock_symbols) < 2: 
        sys.exit("Need at least two stock symbols after alignment for cross-sectional evolution.")
    return aligned_dfs_ordered, common_index, stock_symbols

###############################################################################
# 2. SAFE CORRELATION + CACHES + GUARDS #######################################
###############################################################################
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))): return 0.0
    if a.std(ddof=0) < 1e-9 or b.std(ddof=0) < 1e-9: return 0.0
    if len(a) != len(b) or len(a) < 2: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


_HOF_fingerprints: List[str] = []
_HOF_prediction_timeseries: List[np.ndarray] = [] 
_eval_cache: Dict[str, Tuple[float, float, Optional[np.ndarray]]] = {} # Added Optional[np.ndarray] for raw_preds


def _uses_feature_vector(prog: AlphaProgram) -> bool:
    all_ops_map: Dict[str, Op] = {}
    for op_instance in prog.setup + prog.predict_ops + prog.update_ops:
        all_ops_map[op_instance.out] = op_instance

    if FINAL_PREDICTION_VECTOR_NAME not in all_ops_map and \
       FINAL_PREDICTION_VECTOR_NAME not in INITIAL_STATE_VARS and \
       FINAL_PREDICTION_VECTOR_NAME not in FEATURE_VARS:
        return FINAL_PREDICTION_VECTOR_NAME in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES

    q: List[str] = [FINAL_PREDICTION_VECTOR_NAME]
    visited_vars: Set[str] = set()

    while q:
        current_var_name = q.pop()
        if current_var_name in visited_vars: continue
        visited_vars.add(current_var_name)

        if current_var_name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
            return True 

        defining_op = all_ops_map.get(current_var_name)
        if defining_op:
            for input_var_name in defining_op.inputs:
                if input_var_name not in visited_vars:
                    q.append(input_var_name)
    return False


###############################################################################
# 3. PROGRAM EXECUTION & EVALUATION ###########################################
###############################################################################

def _scale_signal_cross_sectionally_for_ic(raw_signal_vector: np.ndarray, method: str) -> np.ndarray:
    if raw_signal_vector.size == 0:
        return raw_signal_vector
    
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
    else: 
        mu = np.nanmean(clean_signal_vector) 
        sd = np.nanstd(clean_signal_vector)
        if sd < 1e-9 : 
            scaled = np.zeros_like(clean_signal_vector)
        else:
            scaled = (clean_signal_vector - mu) / sd
    
    return np.clip(scaled, -1, 1) 


def evaluate(prog: AlphaProgram) -> Tuple[float, float, Optional[np.ndarray]]:
    fp = prog.fingerprint
    if fp in _eval_cache: return _eval_cache[fp]

    if not _uses_feature_vector(prog):
        _eval_cache[fp] = (-float('inf'), 0.0, None) 
        return _eval_cache[fp]

    program_state: Dict[str, Any] = prog.new_state()
    for s_name, s_type in INITIAL_STATE_VARS.items():
        if s_name not in program_state:
            if s_type == "vector": program_state[s_name] = np.zeros(_N_STOCKS)
            elif s_type == "matrix": program_state[s_name] = np.zeros((_N_STOCKS, _N_STOCKS)) # Should be _N_FEATURES or similar if for feature matrix
            else: program_state[s_name] = 0.0

    raw_predictions_for_hof_correlation: List[np.ndarray] = [] 
    daily_ic_values: List[float] = [] 

    # A1: Adjust loop end for eval_lag. Loop up to `len - EVAL_LAG` to allow fetching future returns.
    # The number of evaluation steps will be `len(_COMMON_TIME_INDEX) - EVAL_LAG`.
    # If EVAL_LAG is 0, it means predicting next bar's return (t+1 from t).
    # If EVAL_LAG is 1, it means predicting t+2's return from t.
    # Predictions are made at `_COMMON_TIME_INDEX[t_idx]`.
    # Forward returns are from `_COMMON_TIME_INDEX[t_idx + EVAL_LAG]`.
    # The `ret_fwd` column itself is already `close.pct_change().shift(-1)`, so it's the next bar's return.
    # So, if EVAL_LAG = 0, we use `ret_fwd` at `_COMMON_TIME_INDEX[t_idx]` which is effectively `(Close[t+1]/Close[t])-1`.
    # If EVAL_LAG = 1, we use `ret_fwd` at `_COMMON_TIME_INDEX[t_idx + 1]` which is `(Close[t+2]/Close[t+1])-1`. This is NOT `(Close[t+EVAL_LAG+1]/Close[t+EVAL_LAG])-1`.
    # The current `ret_fwd` definition seems to be for 1-bar forward return.
    # For a true `EVAL_LAG`, `ret_fwd` should be `df["close"].pct_change(periods=EVAL_LAG).shift(-EVAL_LAG)`
    # OR, we adjust how we access `ret_fwd` here.
    # Let's assume `ret_fwd` is always 1-bar fwd. We need to get the return from `t_idx + EVAL_LAG`'s perspective.
    # This means we need the 1-bar return *after* `_COMMON_TIME_INDEX[t_idx + EVAL_LAG]`.
    # So, the actual return time index will be `_COMMON_TIME_INDEX[t_idx + EVAL_LAG]`. The `ret_fwd` at this time point is the return from `t_idx+EVAL_LAG` to `t_idx+EVAL_LAG+1`.
    # The loop should go from 0 up to `len(_COMMON_TIME_INDEX) - 1 - EVAL_LAG`.
    # This ensures that `t_idx + EVAL_LAG` is a valid index for `_COMMON_TIME_INDEX`.
    loop_end_idx = len(_COMMON_TIME_INDEX) - EVAL_LAG -1 # -1 because ret_fwd is shifted by -1
    if loop_end_idx < 0: # Not enough data for any evaluation with this lag
        _eval_cache[fp] = (-float('inf'), 0.0, None)
        return _eval_cache[fp]


    for t_idx in range(loop_end_idx +1): # +1 because range is exclusive at end
        timestamp = _COMMON_TIME_INDEX[t_idx]
        
        features_at_t: Dict[str, Any] = {}
        for feat_name_template in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
            col_name = feat_name_template.replace('_t', '')
            try:
                feat_vec = np.array([_ALIGNED_DFS[sym].loc[timestamp, col_name] for sym in _STOCK_SYMBOLS], dtype=float)
                features_at_t[feat_name_template] = np.nan_to_num(feat_vec, nan=0.0)
            except KeyError: # Should be rare with ffill/bfill
                features_at_t[feat_name_template] = np.zeros(_N_STOCKS, dtype=float) 

        for sc_name in SCALAR_FEATURE_NAMES: 
            if sc_name == "const_1": features_at_t[sc_name] = 1.0
            elif sc_name == "const_neg_1": features_at_t[sc_name] = -1.0

        try:
            raw_predictions_t = prog.eval(features_at_t, program_state, _N_STOCKS)
            if np.any(np.isnan(raw_predictions_t)) or np.any(np.isinf(raw_predictions_t)):
                _eval_cache[fp] = (-float('inf'), 0.0, None) 
                return _eval_cache[fp]
            raw_predictions_for_hof_correlation.append(raw_predictions_t.copy())

            # B1: Early abort checks (Fix #1 - moved up)
            # Check if we have *at least* EARLY_ABORT_BARS predictions accumulated.
            if len(raw_predictions_for_hof_correlation) >= EARLY_ABORT_BARS:
                # Perform the check only once when we hit EARLY_ABORT_BARS
                if len(raw_predictions_for_hof_correlation) == EARLY_ABORT_BARS: 
                    partial_preds_matrix = np.array(raw_predictions_for_hof_correlation) # No slice needed if check is at ==
                    
                    # Check cross-sectional flatness on partial data
                    mean_xs_std_partial = 0.0
                    if partial_preds_matrix.ndim == 2 and partial_preds_matrix.shape[1] > 0:
                        mean_xs_std_partial = np.mean(partial_preds_matrix.std(axis=1, ddof=0))
                    
                    # Check temporal flatness on partial data
                    mean_t_std_partial = 0.0
                    if partial_preds_matrix.ndim == 2 and partial_preds_matrix.shape[0] > 1:
                        mean_t_std_partial = np.mean(partial_preds_matrix.std(axis=0, ddof=0))
                    elif partial_preds_matrix.ndim == 1 and partial_preds_matrix.shape[0] > 1:
                        mean_t_std_partial = partial_preds_matrix.std(ddof=0)

                    if mean_xs_std_partial < EARLY_ABORT_XS_THRESHOLD or \
                       mean_t_std_partial < EARLY_ABORT_T_THRESHOLD:
                        _eval_cache[fp] = (-float('inf'), 0.0, None)
                        return _eval_cache[fp]

            scaled_predictions_t = _scale_signal_cross_sectionally_for_ic(raw_predictions_t, args.scale)
            mean_scaled_t = np.mean(scaled_predictions_t)
            centered_scaled_t = scaled_predictions_t - mean_scaled_t
            processed_for_ic_t = centered_scaled_t

            # A1: Fetch returns from `t_idx + EVAL_LAG` using the `ret_fwd` column.
            # `ret_fwd` at `_COMMON_TIME_INDEX[t_idx + EVAL_LAG]` gives the return from that time to the next.
            return_timestamp_for_ic = _COMMON_TIME_INDEX[t_idx + EVAL_LAG]
            actual_returns_t = np.array([_ALIGNED_DFS[sym].loc[return_timestamp_for_ic, "ret_fwd"] for sym in _STOCK_SYMBOLS], dtype=float)
            
            if np.any(np.isnan(actual_returns_t)): 
                daily_ic_values.append(0.0) 
            else:
                ic_t = _safe_corr(processed_for_ic_t, actual_returns_t)
                daily_ic_values.append(0.0 if np.isnan(ic_t) else ic_t)

        except Exception: 
            _eval_cache[fp] = (-float('inf'), 0.0, None)
            return _eval_cache[fp]

    if not daily_ic_values or not raw_predictions_for_hof_correlation: 
        _eval_cache[fp] = (-float('inf'), 0.0, None)
        return _eval_cache[fp]

    mean_daily_ic_on_processed = float(np.mean(daily_ic_values))
    score = mean_daily_ic_on_processed - PARSIMONY_PENALTY * prog.size / MAX_OPS 
    
    full_raw_predictions_matrix = np.array(raw_predictions_for_hof_correlation)

    # A2: Cross-sectional flatness guard (on full predictions)
    if full_raw_predictions_matrix.ndim == 2 and full_raw_predictions_matrix.shape[1] > 0:
        cross_sectional_stds = full_raw_predictions_matrix.std(axis=1, ddof=0)
        if np.mean(cross_sectional_stds) < XS_FLATNESS_GUARD_THRESHOLD:
            score = -float('inf') 
            _eval_cache[fp] = (score, mean_daily_ic_on_processed, full_raw_predictions_matrix)
            return score, mean_daily_ic_on_processed, full_raw_predictions_matrix
            
    # Existing temporal flatness guard (MODIFICATION: Tighten the "flat-over-time" guard threshold)
    flat_signal_threshold = 1e-2 
    if full_raw_predictions_matrix.ndim == 2 and full_raw_predictions_matrix.shape[0] > 1: 
        time_std_per_stock = full_raw_predictions_matrix.std(axis=0, ddof=0) 
        if np.mean(time_std_per_stock) < flat_signal_threshold: 
            score = -float('inf') 
    elif full_raw_predictions_matrix.ndim == 1 and full_raw_predictions_matrix.shape[0] > 1: 
        if full_raw_predictions_matrix.std(ddof=0) < flat_signal_threshold:
            score = -float('inf')
    # No need to cache and return here if score became -inf, it will be done after HOF corr penalty.

    if _HOF_prediction_timeseries and score > -float('inf'): # Only apply HOF penalty if score is not already -inf
        current_prog_avg_raw_signal_ts = np.mean(full_raw_predictions_matrix, axis=1) if full_raw_predictions_matrix.ndim > 1 and full_raw_predictions_matrix.shape[1] > 0 else full_raw_predictions_matrix.flatten()
        
        high_corrs = []
        for hof_raw_preds_matrix in _HOF_prediction_timeseries: 
            if hof_raw_preds_matrix.shape[0] != current_prog_avg_raw_signal_ts.shape[0]: continue 
            
            hof_avg_raw_signal_ts = np.mean(hof_raw_preds_matrix, axis=1) if hof_raw_preds_matrix.ndim > 1 and hof_raw_preds_matrix.shape[1] > 0 else hof_raw_preds_matrix.flatten()
            
            current_prog_avg_raw_signal_ts_1d = current_prog_avg_raw_signal_ts.reshape(-1)
            hof_avg_raw_signal_ts_1d = hof_avg_raw_signal_ts.reshape(-1)

            if len(current_prog_avg_raw_signal_ts_1d) != len(hof_avg_raw_signal_ts_1d): continue

            if current_prog_avg_raw_signal_ts_1d.std(ddof=0) < 1e-9 or hof_avg_raw_signal_ts_1d.std(ddof=0) < 1e-9: continue
            
            corr_with_hof = abs(_safe_corr(current_prog_avg_raw_signal_ts_1d, hof_avg_raw_signal_ts_1d))
            if not np.isnan(corr_with_hof) and corr_with_hof > CORR_CUTOFF:
                high_corrs.append(corr_with_hof)
        if high_corrs:
            score -= CORR_PENALTY_W * float(np.mean(high_corrs))
            
    _eval_cache[fp] = (score, mean_daily_ic_on_processed, full_raw_predictions_matrix) 
    return score, mean_daily_ic_on_processed, full_raw_predictions_matrix


###############################################################################
# 4. EA OPERATORS (Wrappers) ##################################################
###############################################################################
def _random_prog() -> AlphaProgram:
    return AlphaProgram.random_program(FEATURE_VARS, INITIAL_STATE_VARS, max_total_ops=MAX_OPS)

def _mutate_prog(p: AlphaProgram) -> AlphaProgram:
    return p.mutate(FEATURE_VARS, INITIAL_STATE_VARS, max_total_ops=MAX_OPS)

def _crossover_prog(a: AlphaProgram, b: AlphaProgram) -> AlphaProgram:
    return a.crossover(b)

###############################################################################
# 5. PROGRESS BAR WRAPPER #####################################################
###############################################################################
def _pbar(iterable, *, desc: str, disable: bool):
    if tqdm and not disable:
        return tqdm(iterable, desc=desc, leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    class _DummyPBar:
        def __init__(self, it, **kwargs): self._it = iter(it)
        def __iter__(self): return self
        def __next__(self): return next(self._it)
        def update(self, *_args, **_kwargs): pass
        def close(self): pass
        def set_postfix_str(self, s): pass
    return _DummyPBar(iterable)

###############################################################################
# 6. EVOLVE LOOP ##############################################################
###############################################################################
def evolve() -> List[Tuple[AlphaProgram, float]]:
    _ensure_data_loaded() 
    
    pop: List[AlphaProgram] = [_random_prog() for _ in range(POP_SIZE)]
    global _HOF_prediction_timeseries, _HOF_fingerprints, _eval_cache
    _HOF_prediction_timeseries = []
    _HOF_fingerprints = [] 
    _eval_cache = {} 
    gen_eval_times_history: List[float] = [] # C2: For ETA calculation

    try:
        for gen in range(N_GENERATIONS): 
            t_start_gen = time.perf_counter()
            eval_results: List[Tuple[int, float, float, Optional[np.ndarray]]] = [] 
            pop_fitness_scores = np.full(POP_SIZE, -np.inf)

            bar = _pbar(range(POP_SIZE), desc=f"Gen {gen+1}/{N_GENERATIONS}", disable=args.quiet)
            for i in bar:
                prog = pop[i]
                score, mean_ic_processed, raw_preds_matrix = evaluate(prog)
                eval_results.append((i, score, mean_ic_processed, raw_preds_matrix))
                pop_fitness_scores[i] = score
                if tqdm and not args.quiet:
                    valid_scores = pop_fitness_scores[pop_fitness_scores > -np.inf]
                    if valid_scores.size > 0:
                        bar.set_postfix_str(f"BestFit: {np.max(valid_scores):.4f}")
                    else:
                        bar.set_postfix_str(f"BestFit: N/A")
            
            gen_eval_time = time.perf_counter() - t_start_gen
            if gen_eval_time > 0: # Avoid division by zero if gen time is somehow 0
                gen_eval_times_history.append(gen_eval_time) # C2
            eval_results.sort(key=lambda x: x[1], reverse=True) 
            
            if not eval_results or eval_results[0][1] <= -float('inf'):
                print(f"Gen {gen+1:3d} | No valid programs found (all scores -inf). Restarting population.")
                pop = [_random_prog() for _ in range(POP_SIZE)] 
                _eval_cache.clear() 
                _HOF_fingerprints.clear()
                _HOF_prediction_timeseries.clear()
                gen_eval_times_history.clear() # C2: Reset history
                continue

            # C2: Calculate ETA
            eta_str = ""
            if gen_eval_times_history: # Guard against empty list (Fix #5 - already good)
                avg_gen_time = np.mean(gen_eval_times_history)
                remaining_gens = N_GENERATIONS - (gen + 1)
                if avg_gen_time > 0 and remaining_gens > 0: # Ensure positive time and remaining gens
                    eta_seconds = remaining_gens * avg_gen_time
                    eta_str = f" | ETA {time.strftime('%Hh%Mm%Ss', time.gmtime(eta_seconds))}"

            best_prog_idx_in_pop, best_fit, best_ic_processed, best_raw_preds_matrix = eval_results[0]
            best_program_this_gen = pop[best_prog_idx_in_pop]

            print(
                f"Gen {gen+1:3d} | BestFit {best_fit:+.4f} | MeanIC {best_ic_processed:+.4f} | Ops {best_program_this_gen.size:2d} | "
                f"EvalTime {gen_eval_time:.1f}s" 
                + eta_str # C2: Added ETA
                + " | " + textwrap.shorten(best_program_this_gen.to_string(), width=50) # Shortened for ETA
            )

            if best_raw_preds_matrix is not None and best_fit > -float('inf'): # Only add to HOF if valid
                fp_best = best_program_this_gen.fingerprint
                if fp_best not in _HOF_fingerprints:
                    _HOF_fingerprints.append(fp_best)
                    _HOF_prediction_timeseries.append(best_raw_preds_matrix) 
                    
                    if len(_HOF_fingerprints) > DUPLICATE_HOF_SZ:
                        _HOF_fingerprints.pop(0)
                        _HOF_prediction_timeseries.pop(0)
            
            new_pop: List[AlphaProgram] = []
            elites_added_fingerprints = set()
            for i_res, score_res, ic_res, _ in eval_results: 
                if score_res <= -float('inf'): continue 
                prog_candidate = pop[i_res]
                fp_cand = prog_candidate.fingerprint
                if fp_cand not in elites_added_fingerprints:
                    new_pop.append(prog_candidate.copy())
                    elites_added_fingerprints.add(fp_cand)
                if len(new_pop) >= ELITE_KEEP:
                    break
            
            if not new_pop and eval_results and eval_results[0][1] > -float('inf'): 
                 new_pop.append(pop[eval_results[0][0]].copy())


            while len(new_pop) < POP_SIZE: 
                valid_indices = [k_idx for k_idx, s_k in enumerate(pop_fitness_scores) if s_k > -np.inf]
                
                if not valid_indices: 
                    new_pop.extend([_random_prog() for _ in range(POP_SIZE - len(new_pop))])
                    break 

                num_to_sample = TOURNAMENT_K * 2
                if len(valid_indices) < num_to_sample : # Allow sampling with replacement if not enough unique options
                    tournament_indices_pool = random.choices(valid_indices, k=num_to_sample)
                else:
                    tournament_indices_pool = random.sample(valid_indices, num_to_sample)
                
                parent1_idx = max(tournament_indices_pool[:TOURNAMENT_K], key=lambda i_tour: pop_fitness_scores[i_tour])
                parent2_idx = max(tournament_indices_pool[TOURNAMENT_K:], key=lambda i_tour: pop_fitness_scores[i_tour])
                
                parent_a, parent_b = pop[parent1_idx], pop[parent2_idx]
                
                child: AlphaProgram
                if random.random() < P_CROSS: child = _crossover_prog(parent_a, parent_b)
                else: child = parent_a.copy() if random.random() < 0.5 else parent_b.copy()
                if random.random() < P_MUT: child = _mutate_prog(child)
                new_pop.append(child)
            pop = new_pop
    except KeyboardInterrupt:
        print("\n[Ctrl‑C] Evolution stopped early. Processing current generation...")

    final_eval_results: List[Tuple[AlphaProgram, float, float]] = [] 
    processed_fps: set[str] = set()
    print("\nEvaluating final population for top unique programs...")
    bar = _pbar(pop, desc="Final Eval", disable=args.quiet)
    for prog_final in bar:
        fp_final = prog_final.fingerprint
        if fp_final in processed_fps: continue
        score_final, ic_final_processed, _ = evaluate(prog_final) # Re-evaluate for consistent metrics if needed
        if score_final > -float('inf'): 
            final_eval_results.append((prog_final, score_final, ic_final_processed))
        processed_fps.add(fp_final)
        
    final_eval_results.sort(key=lambda x: x[1], reverse=True)

    if KEEP_DUPES_IN_HOF_CONFIG:
        top_programs_with_ic_temp = []
        for prog_cand, score_cand, ic_proc_cand in final_eval_results:
            if score_cand > -float('inf'): 
                top_programs_with_ic_temp.append((prog_cand, ic_proc_cand))
            if len(top_programs_with_ic_temp) >= DUPLICATE_HOF_SZ:
                break
        top_programs_with_ic = top_programs_with_ic_temp
    else:
        unique_progs_dict: Dict[str, Tuple[AlphaProgram, float]] = {}
        for prog_candidate, score_candidate, ic_proc_candidate in final_eval_results:
            if score_candidate <= -float('inf'): 
                continue
            
            fp_candidate = prog_candidate.fingerprint
            if fp_candidate not in unique_progs_dict:
                unique_progs_dict[fp_candidate] = (prog_candidate, ic_proc_candidate)
            
            if len(unique_progs_dict) >= DUPLICATE_HOF_SZ: 
                break
        top_programs_with_ic = list(unique_progs_dict.values())

    return top_programs_with_ic

###############################################################################
# 7. ENTRY‑POINT ##############################################################
###############################################################################
if __name__ == "__main__":
    _ensure_data_loaded() 
    print(f"Starting evolution with configuration: {args}")
    top_evolved_alphas = evolve()
    if top_evolved_alphas:
        output_filename = "evolved_top_alphas_standalone.pkl"
        with open(output_filename, "wb") as fh:
            pickle.dump(top_evolved_alphas, fh) 
        print(f"\nSaved top {len(top_evolved_alphas)} programs to {output_filename} (when run standalone)")
    print("\n================ FINAL TOP EVOLVED ALPHAS (Sorted by Fitness Score) ================")
    for i, (prog, ic_processed) in enumerate(top_evolved_alphas, 1):
        print(
            f"#{i:02d} | MeanIC (proc) {ic_processed:+.4f} | Ops {prog.size:3d}\n   " 
            + textwrap.shorten(prog.to_string(), width=1000)
            + "\n"
        )