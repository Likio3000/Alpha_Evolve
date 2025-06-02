from __future__ import annotations
import os
import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict
import pandas as pd
import numpy as np # Though not directly used here, often useful with pandas

# Module-level state for loaded data
_ALIGNED_DFS: Optional[OrderedDictType[str, pd.DataFrame]] = None
_COMMON_TIME_INDEX: Optional[pd.DatetimeIndex] = None
_STOCK_SYMBOLS: Optional[List[str]] = None
_N_STOCKS: Optional[int] = None
_EVAL_LAG_CACHE: int = 1 # Default, will be set by initialize_data

def _rolling_features_individual_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in (5, 10, 20, 30):
        df[f"ma{w}"] = df["close"].rolling(w, min_periods=1).mean()
        df[f"vol{w}"] = df["close"].rolling(w, min_periods=1).std(ddof=0)
    df["range"] = df["high"] - df["low"]
    df["ret_fwd"] = df["close"].pct_change(periods=1).shift(-1) # Shift by -1 for ret_fwd
    return df

def _load_and_align_data_internal(data_dir_param: str, strategy_param: str, min_common_points_param: int, eval_lag: int) -> Tuple[OrderedDictType[str, pd.DataFrame], pd.DatetimeIndex, List[str]]:
    raw_dfs: Dict[str, pd.DataFrame] = {}
    for csv_file in glob.glob(os.path.join(data_dir_param, "*.csv")):
        try:
            df = pd.read_csv(csv_file)
            if 'time' not in df.columns:
                # print(f"Skipping {csv_file}: 'time' column missing.")
                continue
            df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
            df = df.dropna(subset=['time']).sort_values("time").set_index("time")
            if df.empty:
                # print(f"Skipping {csv_file}: empty after time processing.")
                continue
            
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                # print(f"Skipping {csv_file}: missing one of {required_cols}.")
                continue

            df_with_features = _rolling_features_individual_df(df)
            raw_dfs[Path(csv_file).stem] = df_with_features.dropna() # Drop NaNs from rolling features
        except Exception as e:
            # print(f"Error processing {csv_file}: {e}")
            continue

    if not raw_dfs:
        sys.exit(f"No valid CSV data loaded from {data_dir_param}. Ensure files have 'time' and OHLCV columns.")

    if strategy_param == 'specific_long_10k':
        min_len_for_long = min_common_points_param
        raw_dfs = {sym: df for sym, df in raw_dfs.items() if len(df) >= min_len_for_long}
        if len(raw_dfs) < 2:
             sys.exit(f"Not enough long files (>= {min_len_for_long} data points) found for 'specific_long_10k' strategy. Found: {len(raw_dfs)}")

    common_index: Optional[pd.DatetimeIndex] = None
    for sym_name, df_sym in raw_dfs.items(): # Iterate to find common index
        if df_sym.index.has_duplicates:
            print(f"Warning: Duplicate timestamps found in {sym_name}. Keeping first.")
            df_sym = df_sym[~df_sym.index.duplicated(keep='first')]
            raw_dfs[sym_name] = df_sym

        if common_index is None: common_index = df_sym.index
        else: common_index = common_index.intersection(df_sym.index)
    
    # The number of points needed for evaluation is min_common_points_param.
    # The actual data slice needs to be longer by eval_lag to calculate the final forward returns.
    required_length_for_data_slice = min_common_points_param + eval_lag
    
    if common_index is None or len(common_index) < required_length_for_data_slice:
        sys.exit(f"Not enough common history across all symbols. Need {required_length_for_data_slice} (for {min_common_points_param} eval steps + lag {eval_lag}), got {len(common_index) if common_index is not None else 0}).")

    # Truncate common_index if using 'common_1200' or 'specific_long_10k' (unless 'full_overlap')
    if strategy_param == 'common_1200' or strategy_param == 'specific_long_10k':
        # We need `min_common_points_param` for the evaluation period,
        # plus `eval_lag` more for the forward returns needed by the last evaluation point.
        num_points_to_keep_in_slice = min_common_points_param + eval_lag
        if len(common_index) > num_points_to_keep_in_slice:
            common_index = common_index[-num_points_to_keep_in_slice:]
    
    # Now `common_index` is the definitive index for all dataframes.
    # Its length will be `min_common_points_param + eval_lag` (or longer if 'full_overlap' and data allows).

    aligned_dfs_ordered = OrderedDict()
    symbols_to_keep = sorted(raw_dfs.keys())

    for sym in symbols_to_keep:
        df_sym = raw_dfs[sym]
        # Reindex to common_index. ffill/bfill to handle any gaps from individual symbol histories not perfectly matching.
        reindexed_df = df_sym.reindex(common_index).ffill().bfill()
        
        # Final check for NaNs after alignment (should be rare if common_index logic is robust)
        if reindexed_df.isnull().values.any():
             print(f"Warning: DataFrame for {sym} still contains NaNs after ffill/bfill on common_index. This might indicate issues with source data or alignment logic.")
             # Potentially drop this symbol or handle NaNs further, for now, we proceed.
        aligned_dfs_ordered[sym] = reindexed_df

    stock_symbols = list(aligned_dfs_ordered.keys())
    if len(stock_symbols) < 2: # Need at least 2 for cross-sectional
        sys.exit("Need at least two stock symbols after alignment for cross-sectional evolution.")
        
    return aligned_dfs_ordered, common_index, stock_symbols

def initialize_data(data_dir: str, strategy: str, min_common_points: int, eval_lag: int):
    global _ALIGNED_DFS, _COMMON_TIME_INDEX, _STOCK_SYMBOLS, _N_STOCKS, _EVAL_LAG_CACHE
    if _ALIGNED_DFS is not None:
        print("Data already initialized.")
        return

    print(f"Loading and aligning data: dir='{data_dir}', strategy='{strategy}', min_points='{min_common_points}', eval_lag='{eval_lag}'")
    
    _ALIGNED_DFS, _COMMON_TIME_INDEX, _STOCK_SYMBOLS = _load_and_align_data_internal(
        data_dir, strategy, min_common_points, eval_lag
    )
    _N_STOCKS = len(_STOCK_SYMBOLS)
    _EVAL_LAG_CACHE = eval_lag

    print(f"Data initialized: {_N_STOCKS} symbols, {_COMMON_TIME_INDEX.size if _COMMON_TIME_INDEX is not None else 0} common time steps.")
    if _COMMON_TIME_INDEX is not None:
         print(f"Data spans from {_COMMON_TIME_INDEX.min()} to {_COMMON_TIME_INDEX.max()}.")


def get_aligned_dfs() -> OrderedDictType[str, pd.DataFrame]:
    if _ALIGNED_DFS is None:
        raise RuntimeError("Data not initialized. Call initialize_data() first.")
    return _ALIGNED_DFS

def get_common_time_index() -> pd.DatetimeIndex:
    if _COMMON_TIME_INDEX is None:
        raise RuntimeError("Data not initialized. Call initialize_data() first.")
    return _COMMON_TIME_INDEX

def get_stock_symbols() -> List[str]:
    if _STOCK_SYMBOLS is None:
        raise RuntimeError("Data not initialized. Call initialize_data() first.")
    return _STOCK_SYMBOLS

def get_n_stocks() -> int:
    if _N_STOCKS is None:
        raise RuntimeError("Data not initialized. Call initialize_data() first.")
    return _N_STOCKS

def get_eval_lag() -> int:
    # _EVAL_LAG_CACHE is set during initialize_data
    return _EVAL_LAG_CACHE