from __future__ import annotations
import logging
from typing import Dict, List, Tuple, OrderedDict as OrderedDictType
from alpha_evolve.utils.data_loading import DataLoadError, align_and_prune
import pandas as pd
from alpha_evolve.utils.data_loading import load_symbol_dfs_from_dir
from alpha_evolve.utils.features import compute_basic_features
from alpha_evolve.utils.cache import (
    compute_align_cache_key,
    load_aligned_bundle_from_cache,
    save_aligned_bundle_to_cache,
)
# numpy is used by pandas operations, but not directly called here often.


def _rolling_features_individual_df_bt(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper to use the shared feature builder.

    ``ret_fwd`` is recomputed after alignment; we purposefully omit it here.
    """
    return compute_basic_features(df)


def load_and_align_data_for_backtest(
    data_dir_param: str,
    strategy_param: str,
    min_common_points_param: int,
    eval_lag: int = 1,
) -> Tuple[OrderedDictType[str, pd.DataFrame], pd.DatetimeIndex, List[str]]:
    """Load per-symbol data, align indices and return a bundle for backtesting.

    Parameters
    ----------
    data_dir_param : str
        Path to a directory containing one file per symbol. Each file is
        loaded and augmented with rolling features before alignment.
    strategy_param : str
        Name of the backtest strategy. The strategy determines how the
        alignment is performed and may introduce additional filtering, e.g.
        the ``specific_long_10k`` strategy requires enough long files.
    min_common_points_param : int
        Minimum number of overlapping data points required across symbols
        after alignment.
    eval_lag : int, optional
        Forward-return evaluation lag, by default ``1``.

    Returns
    -------
    Tuple[OrderedDict[str, pandas.DataFrame], pandas.DatetimeIndex, List[str]]
        A mapping of stock symbols to their aligned dataframes, the common
        DatetimeIndex shared by all symbols, and the ordered list of symbols
        participating in the backtest.

    Notes
    -----
    An alignment cache is consulted before any expensive computation. If a
    cached bundle for the given parameters is found, it is returned directly.
    Otherwise, the data are loaded and aligned and the resulting bundle is
    saved to the cache for subsequent calls.

    Raises
    ------
    DataLoadError
        Raised when the ``specific_long_10k`` strategy does not have at least
        two symbols meeting ``min_common_points_param`` or when fewer than two
        symbols remain after alignment.
    """
    # Cache key
    key = compute_align_cache_key(
        data_dir=data_dir_param,
        feature_fn_name="compute_basic_features",
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

        if strategy_param == "specific_long_10k":
            raw_dfs = {
                sym: df
                for sym, df in raw_dfs.items()
                if len(df) >= min_common_points_param
            }
            if len(raw_dfs) < 2:
                raise DataLoadError(
                    f"Not enough long files (>= {min_common_points_param} data points) for 'specific_long_10k' backtest strategy. Found: {len(raw_dfs)}"
                )
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
        raise DataLoadError(
            "Need at least two stock symbols after alignment for backtesting."
        )

    return aligned_dfs_ordered, common_index, stock_symbols
