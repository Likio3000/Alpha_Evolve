from __future__ import annotations
import os
import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict

from config import DEFAULT_CRYPTO_SECTOR_MAPPING, DataConfig
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
        sys.exit(f"No valid CSV data loaded from {data_dir_param}. Ensure files have 'time' and OHLCV columns.")

    if strategy_param == 'specific_long_10k':
        min_len_for_long = min_common_points_param
        raw_dfs = {sym: df for sym, df in raw_dfs.items() if len(df) >= min_len_for_long}
        if len(raw_dfs) < 2:
             sys.exit(f"Not enough long files (>= {min_len_for_long} data points) found for 'specific_long_10k' strategy. Found: {len(raw_dfs)}")

    common_index: Optional[pd.DatetimeIndex] = None
    for sym_name, df_sym in raw_dfs.items(): # Iterate to find common index
        if df_sym.index.has_duplicates:
            logger.warning(
                "Duplicate timestamps found in %s. Keeping first.", sym_name
            )
            df_sym = df_sym[~df_sym.index.duplicated(keep='first')]
            raw_dfs[sym_name] = df_sym

        if common_index is None:
            common_index = df_sym.index
        else:
            common_index = common_index.intersection(df_sym.index)
    
    # The number of points needed for evaluation is min_common_points_param.
    # The actual data slice needs to be longer by eval_lag to calculate the final forward returns.
    required_length_for_data_slice = min_common_points_param + eval_lag

    if common_index is None or len(common_index) < required_length_for_data_slice:
        # Attempt to prune symbols with the shortest overlapping window until requirement is met.
        # Heuristic: iteratively drop the symbol limiting the intersection (latest start or earliest end).
        def _compute_bounds(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp, Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]]:
            bounds: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
            starts: List[pd.Timestamp] = []
            ends: List[pd.Timestamp] = []
            for s, dfx in dfs.items():
                st = dfx.index[0]
                en = dfx.index[-1]
                bounds[s] = (st, en)
                starts.append(st)
                ends.append(en)
            return max(starts), min(ends), bounds

        if not raw_dfs:
            sys.exit("No data after initial load.")

        inter_start, inter_end, bounds = _compute_bounds(raw_dfs)
        dropped: List[str] = []

        def _intersection_len(start, end) -> int:
            if start >= end:
                return 0
            return len(pd.DatetimeIndex(sorted(set(raw_dfs[next(iter(raw_dfs))].index)).intersection(pd.date_range(start, end, freq=None))))

        # Loop: drop limiting symbols until intersection long enough or not enough symbols remain
        while True:
            # Recompute common index directly from current raw_dfs
            current_common: Optional[pd.DatetimeIndex] = None
            for dfv in raw_dfs.values():
                current_common = dfv.index if current_common is None else current_common.intersection(dfv.index)
            cur_len = 0 if current_common is None else len(current_common)
            if cur_len >= required_length_for_data_slice:
                common_index = current_common
                break
            if len(raw_dfs) <= 2:
                # Give up – not enough symbols to form cross-section
                sys.exit(
                    f"Not enough common history across all symbols. Need {required_length_for_data_slice} (for {min_common_points_param} eval steps + lag {eval_lag}), "
                    f"got {cur_len}). After pruning {len(dropped)} symbols, only {len(raw_dfs)} remain."
                )

            # Identify candidates causing tight bounds
            inter_start, inter_end, bounds = _compute_bounds(raw_dfs)
            # Find symbols with latest start and earliest end
            latest_start_sym = max(bounds.items(), key=lambda kv: kv[1][0])[0]
            earliest_end_sym = min(bounds.items(), key=lambda kv: kv[1][1])[0]

            # Evaluate which drop improves intersection length more
            def _len_if_drop(sym: str) -> int:
                tmp_common: Optional[pd.DatetimeIndex] = None
                for s, dfx in raw_dfs.items():
                    if s == sym:
                        continue
                    tmp_common = dfx.index if tmp_common is None else tmp_common.intersection(dfx.index)
                return 0 if tmp_common is None else len(tmp_common)

            len_drop_start = _len_if_drop(latest_start_sym)
            len_drop_end = _len_if_drop(earliest_end_sym)
            drop_sym = latest_start_sym if len_drop_start >= len_drop_end else earliest_end_sym
            dropped.append(drop_sym)
            raw_dfs.pop(drop_sym, None)

        if dropped:
            logger.info(
                "Pruned %d symbols with shortest overlap to satisfy required length (%d). Remaining symbols: %d. Dropped: %s",
                len(dropped), required_length_for_data_slice, len(raw_dfs), ", ".join(dropped)
            )

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

        # Ensure forward return matches the configured evaluation horizon.
        # We recompute 'ret_fwd' here so that evaluation at t uses the cumulative
        # return over `eval_lag` bars ending at t+eval_lag.
        try:
            reindexed_df["ret_fwd"] = (
                reindexed_df["close"].pct_change(periods=eval_lag).shift(-eval_lag)
            )
        except Exception:
            # Fall back to 1-step forward return if something goes wrong
            reindexed_df["ret_fwd"] = reindexed_df["close"].pct_change(periods=1).shift(-1)
        
        # Final check for NaNs after alignment. It's expected that the last
        # `eval_lag` rows have NaN in `ret_fwd` (shifted forward returns).
        # Ignore those when deciding whether to warn.
        if eval_lag > 0 and "ret_fwd" in reindexed_df.columns and len(reindexed_df) >= eval_lag:
            tail_ok_mask = pd.DataFrame(False, index=reindexed_df.index, columns=reindexed_df.columns)
            tail_ok_mask.loc[reindexed_df.index[-eval_lag:], "ret_fwd"] = True
            problematic_nans = reindexed_df.isna() & ~tail_ok_mask
        else:
            problematic_nans = reindexed_df.isna()

        if problematic_nans.values.any():
            logger.warning(
                "DataFrame for %s still contains NaNs after alignment (excluding expected tail NaNs in ret_fwd).",
                sym,
            )
            # Potentially drop this symbol or handle NaNs further, for now, we proceed.
        aligned_dfs_ordered[sym] = reindexed_df

    stock_symbols = list(aligned_dfs_ordered.keys())
    if len(stock_symbols) < 2: # Need at least 2 for cross-sectional
        sys.exit("Need at least two stock symbols after alignment for cross-sectional evolution.")
        
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
