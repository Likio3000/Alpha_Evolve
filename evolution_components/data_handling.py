from __future__ import annotations
import os
import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict

from config import DEFAULT_CRYPTO_SECTOR_MAPPING, DataConfig
from utils.data_loading_common import DataDiagnostics, DataLoadError, align_and_prune
import numpy as np
import pandas as pd
import logging
from alpha_framework.alpha_framework_types import (
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    SCALAR_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

# Module-level state for loaded data
_ALIGNED_DFS: Optional[OrderedDictType[str, pd.DataFrame]] = None
_COMMON_TIME_INDEX: Optional[pd.DatetimeIndex] = None
_STOCK_SYMBOLS: Optional[List[str]] = None
_N_STOCKS: Optional[int] = None
_EVAL_LAG_CACHE: int = 1 # Default, will be set by initialize_data
_FEATURE_CACHE: Dict[pd.Timestamp, Dict[str, np.ndarray]] = {}
_DATA_DIAGNOSTICS: Optional[DataDiagnostics] = None

def _rolling_features_individual_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in (5, 10, 20, 30):
        df[f"ma{w}"] = df["close"].rolling(w, min_periods=1).mean()
        df[f"vol{w}"] = df["close"].rolling(w, min_periods=1).std(ddof=0)
    df["range"] = df["high"] - df["low"]
    df["ret_1d"] = df["close"].pct_change().fillna(0.0)
    df["range_rel"] = (df["high"] - df["low"]) / df["close"]
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
        except Exception:
            # print(f"Error processing {csv_file}: {e}")
            continue

    if not raw_dfs:
        raise DataLoadError(f"No valid CSV data loaded from {data_dir_param}. Ensure files have 'time' and OHLCV columns.")

    if strategy_param == 'specific_long_10k':
        min_len_for_long = min_common_points_param
        raw_dfs = {sym: df for sym, df in raw_dfs.items() if len(df) >= min_len_for_long}
        if len(raw_dfs) < 2:
             raise DataLoadError(f"Not enough long files (>= {min_len_for_long} data points) found for 'specific_long_10k' strategy. Found: {len(raw_dfs)}")
    # Use shared alignment + pruning
    bundle = align_and_prune(raw_dfs, strategy_param, min_common_points_param, eval_lag, logger)
    aligned_dfs_ordered = bundle.aligned_dfs
    common_index = bundle.common_index
    stock_symbols = bundle.symbols
    global _DATA_DIAGNOSTICS
    _DATA_DIAGNOSTICS = bundle.diagnostics

    if len(stock_symbols) < 2: # Need at least 2 for cross-sectional
        raise DataLoadError("Need at least two stock symbols after alignment for cross-sectional evolution.")
    
    return aligned_dfs_ordered, common_index, stock_symbols

def initialize_data(data_dir: str, strategy: str, min_common_points: int, eval_lag: int):
    global _ALIGNED_DFS, _COMMON_TIME_INDEX, _STOCK_SYMBOLS, _N_STOCKS, _EVAL_LAG_CACHE
    if _ALIGNED_DFS is not None:
        logger.info("Data already initialized.")
        return

    logger.info(
        "Loading and aligning data: dir='%s', strategy='%s', min_points='%s', eval_lag='%s'",
        data_dir,
        strategy,
        min_common_points,
        eval_lag,
    )
    
    _ALIGNED_DFS, _COMMON_TIME_INDEX, _STOCK_SYMBOLS = _load_and_align_data_internal(
        data_dir, strategy, min_common_points, eval_lag
    )
    _N_STOCKS = len(_STOCK_SYMBOLS)
    _EVAL_LAG_CACHE = eval_lag
    _FEATURE_CACHE.clear()

    if _COMMON_TIME_INDEX is not None:
        sector_groups_vec = get_sector_groups(_STOCK_SYMBOLS)
        for ts in _COMMON_TIME_INDEX:
            _FEATURE_CACHE[ts] = get_features_at_time(
                ts, _ALIGNED_DFS, _STOCK_SYMBOLS, sector_groups_vec
            )

    logger.info(
        "Data initialized: %s symbols, %s common time steps.",
        _N_STOCKS,
        _COMMON_TIME_INDEX.size if _COMMON_TIME_INDEX is not None else 0,
    )
    if _COMMON_TIME_INDEX is not None:
        logger.info(
            "Data spans from %s to %s.",
            _COMMON_TIME_INDEX.min(),
            _COMMON_TIME_INDEX.max(),
        )


def get_data_diagnostics():
    return _DATA_DIAGNOSTICS


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




def _extract_token(sym: str) -> str:
    token = sym.split("_", 1)[-1]
    token = token.split(",", 1)[0]
    token = token.upper()
    for suf in ("USDT", "USDC", "USD"):
        if token.endswith(suf):
            token = token[: -len(suf)]
            break
    return token


def get_sector_groups(
    symbols: Optional[List[str]] = None,
    mapping: Dict[str, int] | None = None,
    cfg: Optional[DataConfig] = None,
) -> np.ndarray:
    """Return sector IDs for provided symbols.

    If ``symbols`` is ``None`` the currently loaded stock symbols are used.
    ``mapping`` allows overriding the default token → sector mapping.
    """

    if symbols is None:
        if _STOCK_SYMBOLS is None:
            raise RuntimeError("Data not initialized. Call initialize_data() first.")
        symbols = _STOCK_SYMBOLS

    if mapping is not None:
        sector_map = mapping
    elif cfg is not None:
        sector_map = cfg.sector_mapping
    else:
        sector_map = DEFAULT_CRYPTO_SECTOR_MAPPING

    groups: List[int] = []
    for s in symbols:
        token = _extract_token(s)
        sector = sector_map.get(token, -1)
        groups.append(sector)

    return np.array(groups, dtype=int)


def get_data_splits(train_points: int, val_points: int, test_points: int) -> Tuple[
    OrderedDictType[str, pd.DataFrame],
    OrderedDictType[str, pd.DataFrame],
    OrderedDictType[str, pd.DataFrame],
]:
    """Return train/validation/test slices of the aligned data.

    The number of *points* refers to evaluation steps. Because the common index
    includes ``eval_lag`` extra rows to compute the forward returns, each split
    will contain ``points + eval_lag`` rows from the underlying data.
    """

    if _ALIGNED_DFS is None or _COMMON_TIME_INDEX is None:
        raise RuntimeError("Data not initialized. Call initialize_data() first.")

    total_eval_steps = len(_COMMON_TIME_INDEX) - _EVAL_LAG_CACHE
    required = train_points + val_points + test_points
    if required > total_eval_steps:
        raise ValueError(
            f"Requested split of {required} eval steps exceeds available {total_eval_steps}."
        )

    slices = []
    start = 0
    for size in (train_points, val_points, test_points):
        idx_slice = _COMMON_TIME_INDEX[start : start + size + _EVAL_LAG_CACHE]
        split_dfs = OrderedDict({sym: df.loc[idx_slice] for sym, df in _ALIGNED_DFS.items()})
        slices.append(split_dfs)
        start += size

    return tuple(slices)  # type: ignore[return-value]

def get_features_at_time(timestamp, aligned_dfs, stock_symbols, sector_groups_vec):
    """Return feature dict for a given timestamp.

    Parameters
    ----------
    timestamp : Any
        Timestamp at which to collect features.
    aligned_dfs : Mapping[str, pd.DataFrame]
        Aligned data frames keyed by symbol.
    stock_symbols : Sequence[str]
        Symbols to pull data for.
    sector_groups_vec : np.ndarray
        Precomputed sector id vector for ``stock_symbols``.
    """
    features_at_t = {}
    n_stocks = len(stock_symbols)

    for feat_name_template in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
        if feat_name_template == "sector_id_vector":
            features_at_t[feat_name_template] = sector_groups_vec
            continue
        col_name = feat_name_template.replace("_t", "")
        try:
            vec = np.array([
                aligned_dfs[sym].loc[timestamp, col_name] for sym in stock_symbols
            ], dtype=float)
            features_at_t[feat_name_template] = np.nan_to_num(
                vec, nan=0.0, posinf=0.0, neginf=0.0
            )
        except KeyError:
            features_at_t[feat_name_template] = np.zeros(n_stocks, dtype=float)
        except Exception:
            features_at_t[feat_name_template] = np.zeros(n_stocks, dtype=float)

    for sc_name in SCALAR_FEATURE_NAMES:
        if sc_name == "const_1":
            features_at_t[sc_name] = 1.0
        elif sc_name == "const_neg_1":
            features_at_t[sc_name] = -1.0

    return features_at_t


def get_features_cached(timestamp) -> Dict[str, np.ndarray]:
    """Return cached features for ``timestamp``."""
    if timestamp in _FEATURE_CACHE:
        return _FEATURE_CACHE[timestamp]
    if _ALIGNED_DFS is None or _STOCK_SYMBOLS is None:
        raise RuntimeError("Data not initialized. Call initialize_data() first.")
    sector_groups_vec = get_sector_groups(_STOCK_SYMBOLS)
    features = get_features_at_time(timestamp, _ALIGNED_DFS, _STOCK_SYMBOLS, sector_groups_vec)
    _FEATURE_CACHE[timestamp] = features
    return features


def clear_feature_cache() -> None:
    """Clear the precomputed feature cache."""
    _FEATURE_CACHE.clear()
