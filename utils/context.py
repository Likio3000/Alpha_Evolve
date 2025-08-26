from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from utils.data_loading_common import DataBundle, align_and_prune, load_symbol_dfs_from_dir


@dataclass(frozen=True)
class EvalContext:
    bundle: DataBundle
    sector_ids: np.ndarray
    eval_lag: int


def make_eval_context_from_globals(dh_module) -> EvalContext:
    """Build an EvalContext from the currently initialized evolution data_handling globals."""
    aligned = dh_module.get_aligned_dfs()
    symbols = dh_module.get_stock_symbols()
    common_index = dh_module.get_common_time_index()
    eval_lag = dh_module.get_eval_lag()
    # Build a minimal DataBundle wrapper
    bundle = DataBundle(aligned_dfs=aligned, common_index=common_index, symbols=symbols,
                        diagnostics=dh_module.get_data_diagnostics())
    sector_ids = dh_module.get_sector_groups(symbols)
    return EvalContext(bundle=bundle, sector_ids=sector_ids, eval_lag=eval_lag)


def make_eval_context_from_dir(
    *,
    data_dir: str,
    strategy: str,
    min_common_points: int,
    eval_lag: int,
    dh_module,
    sector_mapping: dict | None = None,
) -> EvalContext:
    """Load, align, and construct an EvalContext directly from a data directory.

    Uses the evolution feature builder present in the provided ``dh_module``.
    """
    feature_fn = getattr(dh_module, "_rolling_features_individual_df")
    raw_dfs = load_symbol_dfs_from_dir(data_dir, feature_fn)
    bundle = align_and_prune(raw_dfs, strategy, min_common_points, eval_lag, logger=_NullLogger())
    if sector_mapping is not None:
        sector_ids = dh_module.get_sector_groups(bundle.symbols, mapping=sector_mapping)
    else:
        sector_ids = dh_module.get_sector_groups(bundle.symbols)
    return EvalContext(bundle=bundle, sector_ids=sector_ids, eval_lag=eval_lag)


class _NullLogger:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop
