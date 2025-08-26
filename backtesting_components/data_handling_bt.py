from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType
from utils.data_loading_common import DataLoadError, align_and_prune
from collections import OrderedDict
import pandas as pd
from utils.data_loading_common import load_symbol_dfs_from_dir
from utils.cache import compute_align_cache_key, load_aligned_bundle_from_cache, save_aligned_bundle_to_cache
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
    # Cache key
    key = compute_align_cache_key(
        data_dir=data_dir_param,
        feature_fn_name="_rolling_features_individual_df_bt",
        strategy=strategy_param,
        min_common_points=min_common_points_param,
        eval_lag=eval_lag,
        include_lag_in_required_length=False,
        fixed_trim_include_lag=False,
    )
    bundle = load_aligned_bundle_from_cache(key)
    if bundle is None:
        raw_dfs: Dict[str, pd.DataFrame] = load_symbol_dfs_from_dir(
            data_dir_param, _rolling_features_individual_df_bt
        )

        if strategy_param == 'specific_long_10k':
            raw_dfs = {sym: df for sym, df in raw_dfs.items() if len(df) >= min_common_points_param}
            if len(raw_dfs) < 2:
                 raise DataLoadError(f"Not enough long files (>= {min_common_points_param} data points) for 'specific_long_10k' backtest strategy. Found: {len(raw_dfs)}")
        # Use shared alignment + pruning (for all strategies)
        bundle = align_and_prune(
            raw_dfs,
            strategy_param,
            min_common_points_param,
            eval_lag,
            logging.getLogger(__name__),
            include_lag_in_required_length=False,
            fixed_trim_include_lag=False,
        )
        save_aligned_bundle_to_cache(key, bundle)
    aligned_dfs_ordered = bundle.aligned_dfs
    common_index = bundle.common_index
    stock_symbols = bundle.symbols

    if len(stock_symbols) < 2:
        raise DataLoadError("Need at least two stock symbols after alignment for backtesting.")
        
    return aligned_dfs_ordered, common_index, stock_symbols
