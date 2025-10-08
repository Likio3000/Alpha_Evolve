from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict

from config import DEFAULT_SECTOR_MAPPING, DataConfig
from utils.data_loading_common import DataDiagnostics, DataLoadError, align_and_prune, load_symbol_dfs_from_dir
from utils.cache import compute_align_cache_key, load_aligned_bundle_from_cache, save_aligned_bundle_to_cache
import numpy as np
import pandas as pd
import logging
from alpha_framework.alpha_framework_types import (
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    SCALAR_FEATURE_NAMES,
)
from utils.features import compute_basic_features

DERIVED_VECTOR_FEATURES = {
    "market_rel_close_t",
    "market_rel_ret1d_t",
    "market_zclose_t",
    "btc_ratio_proxy_t",
    "regime_volatility_t",
    "regime_momentum_t",
    "market_dispersion_t",
    "cross_btc_momentum_t",
    "sector_momentum_diff_t",
}

logger = logging.getLogger(__name__)

# Module-level state for loaded data
_ALIGNED_DFS: Optional[OrderedDictType[str, pd.DataFrame]] = None
_COMMON_TIME_INDEX: Optional[pd.DatetimeIndex] = None
_STOCK_SYMBOLS: Optional[List[str]] = None
_N_STOCKS: Optional[int] = None
_EVAL_LAG_CACHE: int = 1 # Default, will be set by initialize_data
_FEATURE_CACHE: Dict[pd.Timestamp, Dict[str, np.ndarray]] = {}
_DATA_DIAGNOSTICS: Optional[DataDiagnostics] = None
_DEPRECATION_WARNED = False

def _rolling_features_individual_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper that delegates to the shared feature builder.

    ``ret_fwd`` is recomputed after alignment; we purposefully omit it here.
    """
    return compute_basic_features(df)

def _load_and_align_data_internal(data_dir_param: str, strategy_param: str, min_common_points_param: int, eval_lag: int) -> Tuple[OrderedDictType[str, pd.DataFrame], pd.DatetimeIndex, List[str]]:
    # Try cache first
    key = compute_align_cache_key(
        data_dir=data_dir_param,
        feature_fn_name="compute_basic_features",
        strategy=strategy_param,
        min_common_points=min_common_points_param,
        eval_lag=eval_lag,
        include_lag_in_required_length=True,
        fixed_trim_include_lag=True,
    )
    bundle = load_aligned_bundle_from_cache(key)
    if bundle is None:
        raw_dfs: Dict[str, pd.DataFrame] = load_symbol_dfs_from_dir(
            data_dir_param, _rolling_features_individual_df
        )
        if strategy_param == 'specific_long_10k':
            min_len_for_long = min_common_points_param
            raw_dfs = {sym: df for sym, df in raw_dfs.items() if len(df) >= min_len_for_long}
            if len(raw_dfs) < 2:
                 raise DataLoadError(f"Not enough long files (>= {min_len_for_long} data points) found for 'specific_long_10k' strategy. Found: {len(raw_dfs)}")
        # Use shared alignment + pruning
        bundle = align_and_prune(raw_dfs, strategy_param, min_common_points_param, eval_lag, logger)
        save_aligned_bundle_to_cache(key, bundle)
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
        try:
            n_buckets = int(len(np.unique(sector_groups_vec)))
            logger.info("Sector buckets detected: %d (unknowns map to -1)", n_buckets)
        except Exception:
            pass

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
    global _DEPRECATION_WARNED
    if not _DEPRECATION_WARNED:
        try:
            logger.info(
                "[DEPRECATED] data_handling getters will be removed; prefer EvalContext (utils.context)."
            )
        except Exception:
            pass
        _DEPRECATION_WARNED = True
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
        sector_map = DEFAULT_SECTOR_MAPPING

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
    """Return feature dict for a given timestamp."""
    features_at_t: Dict[str, np.ndarray] = {}
    n_stocks = len(stock_symbols)
    zeros = np.zeros(n_stocks, dtype=float)

    column_cache: Dict[str, np.ndarray] = {}

    def _get_column(col_name: str) -> np.ndarray:
        if col_name in column_cache:
            return column_cache[col_name]
        arr = np.zeros(n_stocks, dtype=float)
        for idx, sym in enumerate(stock_symbols):
            try:
                arr[idx] = float(aligned_dfs[sym].loc[timestamp, col_name])
            except Exception:
                arr[idx] = 0.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        column_cache[col_name] = arr
        return arr

    eps = 1e-9
    close_vec = _get_column("closes")
    ret_vec = _get_column("ret1d")
    vol20_vec = _get_column("vol20")
    trend_vec = _get_column("trend_5_20")

    reference_idx = 0
    for idx, sym in enumerate(stock_symbols):
        sym_upper = sym.upper()
        if "BTC" in sym_upper or "SPY" in sym_upper or "^GSPC" in sym_upper:
            reference_idx = idx
            break

    derived_vectors: Dict[str, np.ndarray] = {}
    if close_vec.size:
        mean_close = float(np.mean(close_vec))
        if abs(mean_close) > eps:
            rel_close = (close_vec / mean_close) - 1.0
        else:
            rel_close = np.zeros_like(close_vec)
        derived_vectors["market_rel_close_t"] = np.nan_to_num(rel_close, nan=0.0, posinf=0.0, neginf=0.0)

        std_close = float(np.std(close_vec, ddof=0))
        if std_close > eps:
            zclose = (close_vec - mean_close) / std_close
        else:
            zclose = np.zeros_like(close_vec)
        derived_vectors["market_zclose_t"] = np.nan_to_num(zclose, nan=0.0, posinf=0.0, neginf=0.0)

        ref_close = float(close_vec[reference_idx]) if close_vec.size > reference_idx else 0.0
        if abs(ref_close) > eps:
            btc_ratio = (close_vec / ref_close) - 1.0
        else:
            btc_ratio = np.zeros_like(close_vec)
        derived_vectors["btc_ratio_proxy_t"] = np.nan_to_num(btc_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        median_close = float(np.median(close_vec))
        dispersion = close_vec - median_close
        derived_vectors["market_dispersion_t"] = np.nan_to_num(dispersion, nan=0.0, posinf=0.0, neginf=0.0)

    if ret_vec.size:
        mean_ret = float(np.mean(ret_vec))
        derived_vectors["market_rel_ret1d_t"] = np.nan_to_num(ret_vec - mean_ret, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        derived_vectors["market_rel_ret1d_t"] = zeros

    if vol20_vec.size:
        mean_vol20 = float(np.mean(vol20_vec))
        if abs(mean_vol20) > eps:
            rel_vol = (vol20_vec / mean_vol20) - 1.0
        else:
            rel_vol = np.zeros_like(vol20_vec)
        derived_vectors["regime_volatility_t"] = np.nan_to_num(rel_vol, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        derived_vectors["regime_volatility_t"] = zeros

    if trend_vec.size:
        mean_trend = float(np.mean(trend_vec))
        derived_vectors["regime_momentum_t"] = np.nan_to_num(trend_vec - mean_trend, nan=0.0, posinf=0.0, neginf=0.0)
        ref_trend = float(trend_vec[reference_idx]) if trend_vec.size > reference_idx else 0.0
        cross_btc = trend_vec - ref_trend
        derived_vectors["cross_btc_momentum_t"] = np.nan_to_num(cross_btc, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            sector_diff = np.zeros_like(trend_vec)
            if sector_groups_vec is not None and sector_groups_vec.size == trend_vec.size:
                for sector_id in np.unique(sector_groups_vec):
                    mask = sector_groups_vec == sector_id
                    if not np.any(mask):
                        continue
                    sector_mean = float(np.mean(trend_vec[mask]))
                    sector_diff[mask] = trend_vec[mask] - sector_mean
            derived_vectors["sector_momentum_diff_t"] = np.nan_to_num(sector_diff, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            derived_vectors["sector_momentum_diff_t"] = zeros
    else:
        derived_vectors["regime_momentum_t"] = zeros
        derived_vectors["cross_btc_momentum_t"] = zeros
        derived_vectors["sector_momentum_diff_t"] = zeros

    for feat_name_template in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
        if feat_name_template == "sector_id_vector":
            features_at_t[feat_name_template] = sector_groups_vec
            continue
        derived_vec = derived_vectors.get(feat_name_template)
        if derived_vec is not None:
            features_at_t[feat_name_template] = derived_vec.copy()
            continue
        col_name = feat_name_template.replace("_t", "")
        vec = _get_column(col_name)
        features_at_t[feat_name_template] = vec.copy() if vec.size else zeros.copy()

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
