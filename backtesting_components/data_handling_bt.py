from __future__ import annotations
import os
import glob
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType
from utils.data_loading_common import DataLoadError, align_and_prune
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
        raise DataLoadError(f"No valid CSV data loaded for backtesting from {data_dir_param}.")

    if strategy_param == 'specific_long_10k':
        raw_dfs = {sym: df for sym, df in raw_dfs.items() if len(df) >= min_common_points_param}
        if len(raw_dfs) < 2:
             raise DataLoadError(f"Not enough long files (>= {min_common_points_param} data points) for 'specific_long_10k' backtest strategy. Found: {len(raw_dfs)}")
    # Use shared alignment + pruning
    bundle = align_and_prune(
        raw_dfs,
        strategy_param,
        min_common_points_param,
        eval_lag,
        logging.getLogger(__name__),
        include_lag_in_required_length=False,
        fixed_trim_include_lag=False,
    )
    aligned_dfs_ordered = bundle.aligned_dfs
    common_index = bundle.common_index
    stock_symbols = bundle.symbols

    if len(stock_symbols) < 2:
        raise DataLoadError("Need at least two stock symbols after alignment for backtesting.")
        
    return aligned_dfs_ordered, common_index, stock_symbols
