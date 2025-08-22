from __future__ import annotations
import os
import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict
import pandas as pd
# numpy is used by pandas operations, but not directly called here often.

def _rolling_features_individual_df_bt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in (5, 10, 20, 30):
        df[f"ma{w}"] = df["close"].rolling(w, min_periods=1).mean()
        df[f"vol{w}"] = df["close"].rolling(w, min_periods=1).std(ddof=0)
    df["range"] = df["high"] - df["low"]
    df["ret_1d"] = df["close"].pct_change().fillna(0.0)
    df["range_rel"] = (df["high"] - df["low"]) / df["close"]
    df["ret_fwd"] = df["close"].pct_change(periods=1).shift(-1)
    return df

def load_and_align_data_for_backtest(
    data_dir_param: str,
    strategy_param: str,
    min_common_points_param: int,
    eval_lag: int = 1,
) -> Tuple[OrderedDictType[str, pd.DataFrame], pd.DatetimeIndex, List[str]]:
    raw_dfs: Dict[str, pd.DataFrame] = {}
    for csv_file in glob.glob(os.path.join(data_dir_param, "*.csv")):
        try:
            df = pd.read_csv(csv_file)
            if 'time' not in df.columns:
                continue
            df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
            df = df.dropna(subset=['time']).sort_values("time").set_index("time")
            if df.empty:
                continue
            
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                continue

            df_with_features = _rolling_features_individual_df_bt(df)
            raw_dfs[Path(csv_file).stem] = df_with_features.dropna()
        except Exception:
            # print(f"Error processing {csv_file} for backtest data: {e}") # Optional debug
            continue

    if not raw_dfs:
        sys.exit(f"No valid CSV data loaded for backtesting from {data_dir_param}.")

    if strategy_param == 'specific_long_10k':
        raw_dfs = {sym: df for sym, df in raw_dfs.items() if len(df) >= min_common_points_param}
        if len(raw_dfs) < 2:
             sys.exit(f"Not enough long files (>= {min_common_points_param} data points) for 'specific_long_10k' backtest strategy. Found: {len(raw_dfs)}")

    common_index: Optional[pd.DatetimeIndex] = None
    for sym_name, df_sym in raw_dfs.items():
        if df_sym.index.has_duplicates:
            # print(f"Warning (backtest): Duplicate timestamps found in {sym_name}. Keeping first.") # Optional
            df_sym = df_sym[~df_sym.index.duplicated(keep='first')]
            raw_dfs[sym_name] = df_sym
        if common_index is None:
            common_index = df_sym.index
        else:
            common_index = common_index.intersection(df_sym.index)
    
    # For backtests we also need to ensure there are enough points to compute
    # forward returns for the chosen evaluation lag.
    required_len = min_common_points_param + max(int(eval_lag), 0)
    if common_index is None or len(common_index) < required_len:
        sys.exit(
            f"Not enough common history for backtesting (need at least {required_len}, "
            f"got {len(common_index) if common_index is not None else 0})."
        )

    if strategy_param == 'common_1200':  # fixed lookback window
        # Keep exactly the evaluation steps plus extra bars for the forward return horizon
        need = min_common_points_param + max(int(eval_lag), 0)
        if len(common_index) > need:
            common_index = common_index[-need:]
    # For 'specific_long_10k' and 'full_overlap', use the full common_index found that meets min_common_points_param.

    aligned_dfs_ordered = OrderedDict()
    symbols_to_keep = sorted(list(raw_dfs.keys())) # Ensure we only iterate over symbols present after filtering

    for sym in symbols_to_keep:
        # Ensure symbol still exists in raw_dfs (it should if symbols_to_keep is from raw_dfs.keys())
        if sym not in raw_dfs:
            continue
        
        df_sym = raw_dfs[sym].reindex(common_index).ffill().bfill()

        # Recompute forward returns to match the configured evaluation horizon
        try:
            df_sym["ret_fwd"] = (
                df_sym["close"].pct_change(periods=eval_lag).shift(-eval_lag)
            )
        except Exception:
            df_sym["ret_fwd"] = df_sym["close"].pct_change(periods=1).shift(-1)

        if df_sym.isnull().values.any():
             print(f"Warning (backtest): DataFrame for {sym} contains NaNs after alignment. This might affect backtest results.")
        aligned_dfs_ordered[sym] = df_sym
    
    stock_symbols = list(aligned_dfs_ordered.keys())
    if len(stock_symbols) < 2:
        sys.exit("Need at least two stock symbols after alignment for backtesting.")
        
    return aligned_dfs_ordered, common_index, stock_symbols
