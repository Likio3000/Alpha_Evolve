from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .data_loading import DataBundle, align_and_prune, load_symbol_dfs_from_dir
from .cache import (
    compute_align_cache_key,
    load_aligned_bundle_from_cache,
    save_aligned_bundle_to_cache,
)


@dataclass(frozen=True)
class EvalContext:
    bundle: DataBundle
    sector_ids: np.ndarray
    eval_lag: int
    col_matrix_map: Optional[Dict[str, np.ndarray]] = field(default=None)
    ts_pos: Optional[Dict[pd.Timestamp, int]] = field(default=None)


def make_eval_context_from_globals(dh_module) -> EvalContext:
    """Build an EvalContext from the currently initialized evolution data_handling globals.

    DEPRECATED: Prefer building contexts directly from disk via
    ``make_eval_context_from_dir`` or from a prebuilt ``DataBundle`` using
    ``make_eval_context_from_bundle``. This helper exists for back‑compat only.
    """
    try:
        import logging
        logging.getLogger(__name__).info(
            "[DEPRECATED] make_eval_context_from_globals: prefer context from dir or bundle."
        )
    except Exception:
        pass
    aligned = dh_module.get_aligned_dfs()
    symbols = dh_module.get_stock_symbols()
    common_index = dh_module.get_common_time_index()
    eval_lag = dh_module.get_eval_lag()
    # Build a minimal DataBundle wrapper
    bundle = DataBundle(aligned_dfs=aligned, common_index=common_index, symbols=symbols,
                        diagnostics=dh_module.get_data_diagnostics())
    sector_ids = dh_module.get_sector_groups(symbols)
    return EvalContext(bundle=bundle, sector_ids=sector_ids, eval_lag=eval_lag)


def make_eval_context_from_bundle(
    bundle: DataBundle,
    *,
    eval_lag: int,
    dh_module,
    sector_mapping: dict | None = None,
    precompute_columns: List[str] | None = None,
) -> EvalContext:
    """Construct an EvalContext from an existing DataBundle.

    Useful when composing multiple runs or for tests that already have aligned
    data. Optionally precomputes column matrices and a timestamp→position map.
    """
    if sector_mapping is not None:
        sector_ids = dh_module.get_sector_groups(bundle.symbols, mapping=sector_mapping)
    else:
        sector_ids = dh_module.get_sector_groups(bundle.symbols)

    col_map = None
    ts_pos = None
    if precompute_columns:
        T = len(bundle.common_index)
        N = len(bundle.symbols)
        ts_pos = {ts: i for i, ts in enumerate(bundle.common_index)}
        col_map = {}
        for col in precompute_columns:
            mat = np.zeros((T, N), dtype=float)
            for j, sym in enumerate(bundle.symbols):
                try:
                    series = bundle.aligned_dfs[sym][col].reindex(bundle.common_index)
                    mat[:, j] = np.nan_to_num(series.values, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    mat[:, j] = 0.0
            col_map[col] = mat

    return EvalContext(bundle=bundle, sector_ids=sector_ids, eval_lag=eval_lag, col_matrix_map=col_map, ts_pos=ts_pos)


def make_eval_context_from_dir(
    *,
    data_dir: str,
    strategy: str,
    min_common_points: int,
    eval_lag: int,
    dh_module,
    sector_mapping: dict | None = None,
    precompute_columns: List[str] | None = None,
) -> EvalContext:
    """Load, align, and construct an EvalContext directly from a data directory.

    Uses the evolution feature builder present in the provided ``dh_module``.
    """
    feature_fn = getattr(dh_module, "_rolling_features_individual_df")
    # Try cache
    key = compute_align_cache_key(
        data_dir=data_dir,
        feature_fn_name=getattr(feature_fn, "__name__", "feat"),
        strategy=strategy,
        min_common_points=min_common_points,
        eval_lag=eval_lag,
        include_lag_in_required_length=True,
        fixed_trim_include_lag=True,
    )
    bundle = load_aligned_bundle_from_cache(key)
    if bundle is None:
        raw_dfs = load_symbol_dfs_from_dir(data_dir, feature_fn)
        bundle = align_and_prune(raw_dfs, strategy, min_common_points, eval_lag, logger=_NullLogger())
        save_aligned_bundle_to_cache(key, bundle)
    if sector_mapping is not None:
        sector_ids = dh_module.get_sector_groups(bundle.symbols, mapping=sector_mapping)
    else:
        sector_ids = dh_module.get_sector_groups(bundle.symbols)

    col_map = None
    ts_pos = None
    if precompute_columns:
        T = len(bundle.common_index)
        N = len(bundle.symbols)
        ts_pos = {ts: i for i, ts in enumerate(bundle.common_index)}
        col_map = {}
        for col in precompute_columns:
            mat = np.zeros((T, N), dtype=float)
            for j, sym in enumerate(bundle.symbols):
                try:
                    series = bundle.aligned_dfs[sym][col].reindex(bundle.common_index)
                    mat[:, j] = np.nan_to_num(series.values, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    mat[:, j] = 0.0
            col_map[col] = mat

    return EvalContext(bundle=bundle, sector_ids=sector_ids, eval_lag=eval_lag, col_matrix_map=col_map, ts_pos=ts_pos)


def slice_eval_context(ctx: EvalContext, *, eval_fraction: float | None = None, last_steps: int | None = None) -> EvalContext:
    """Create a shallowly sliced EvalContext limited to a trailing window.

    - If ``last_steps`` is provided, keep the last ``last_steps`` evaluation bars.
    - Otherwise, use ``eval_fraction`` of available evaluation bars (0<frac<=1).

    The resulting context includes the additional ``eval_lag`` rows required by
    consumers that compute forward returns, matching the semantics of
    ``get_data_splits``.
    """
    bundle = ctx.bundle
    eval_lag = int(ctx.eval_lag)
    total = max(0, len(bundle.common_index) - eval_lag)
    if total <= 0:
        return ctx
    if last_steps is None:
        f = float(eval_fraction or 1.0)
        f = max(0.0, min(1.0, f))
        points = max(1, int(round(total * f)))
    else:
        points = max(1, int(last_steps))
    points = min(points, total)
    keep = points + eval_lag
    # Take trailing window
    new_index = bundle.common_index[-keep:]
    new_dfs: Dict[str, pd.DataFrame] = {}
    for sym, df in bundle.aligned_dfs.items():
        try:
            new_dfs[sym] = df.loc[new_index]
        except Exception:
            new_dfs[sym] = df.iloc[-keep:]
    from .data_loading import DataBundle
    new_bundle = DataBundle(
        aligned_dfs=new_dfs,
        common_index=new_index,
        symbols=list(bundle.symbols),
        diagnostics=bundle.diagnostics,
    )
    # Slice precomputed matrices and timestamp map if present
    col_map = None
    ts_pos = None
    if ctx.col_matrix_map is not None:
        col_map = {k: v[-keep:, :].copy() for k, v in ctx.col_matrix_map.items()}
        ts_pos = {ts: i for i, ts in enumerate(new_index)}
    return EvalContext(bundle=new_bundle, sector_ids=ctx.sector_ids, eval_lag=eval_lag, col_matrix_map=col_map, ts_pos=ts_pos)


class _NullLogger:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop
