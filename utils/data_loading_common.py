from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType, Callable
from collections import OrderedDict
import pandas as pd
from utils.errors import DataLoadError


@dataclass
class DataDiagnostics:
    n_symbols_before: int
    n_symbols_after: int
    dropped_symbols: List[str]
    overlap_len: int
    overlap_start: Optional[pd.Timestamp]
    overlap_end: Optional[pd.Timestamp]


@dataclass
class DataBundle:
    aligned_dfs: OrderedDictType[str, pd.DataFrame]
    common_index: pd.DatetimeIndex
    symbols: List[str]
    diagnostics: DataDiagnostics

REQUIRED_OHLC = ("open", "high", "low", "close")


def prepare_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a raw OHLCV dataframe.

    - Requires a "time" column with seconds since epoch or ISO8601; converts to datetime index.
    - Ensures required OHLC columns are present.
    - Sorts by time and removes duplicate timestamps (keeping first).
    """
    if "time" not in df.columns:
        raise DataLoadError("Missing 'time' column")
    df = df.copy()
    # Robust timestamp parsing: handle numeric epoch seconds and ISO8601 strings
    time_col = df["time"]
    try:
        if pd.api.types.is_numeric_dtype(time_col):
            parsed = pd.to_datetime(time_col, unit="s", errors="coerce")
        else:
            parsed = pd.to_datetime(time_col, errors="coerce", utc=False, infer_datetime_format=True)
    except Exception:
        parsed = pd.to_datetime(time_col, errors="coerce")
    df["time"] = parsed
    df = df.dropna(subset=["time"]).sort_values("time").set_index("time")
    for col in REQUIRED_OHLC:
        if col not in df.columns:
            raise DataLoadError(f"Missing required column: {col}")
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    return df


def load_symbol_dfs_from_dir(
    data_dir: str,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Load CSVs from a directory, validate schema, and compute features.

    Returns a mapping {symbol -> prepared_dataframe}. Skips invalid files; if none
    valid, raises DataLoadError.
    """
    import os, glob
    from pathlib import Path
    out: Dict[str, pd.DataFrame] = {}
    for csv_file in glob.glob(os.path.join(data_dir, "*.csv")):
        try:
            raw = pd.read_csv(csv_file)
            base = prepare_ohlcv_df(raw)
            with_feat = feature_fn(base)
            with_feat = with_feat.dropna()
            if not with_feat.empty:
                out[Path(csv_file).stem] = with_feat
        except Exception:
            continue
    if not out:
        raise DataLoadError(f"No valid CSV data loaded from {data_dir}.")
    return out


def align_and_prune(
    raw_dfs: Dict[str, pd.DataFrame],
    strategy_param: str,
    min_common_points_param: int,
    eval_lag: int,
    logger,
    *,
    recompute_ret_fwd: bool = True,
    include_lag_in_required_length: bool = True,
    fixed_trim_include_lag: bool = True,
) -> DataBundle:
    if not raw_dfs:
        raise DataLoadError("No valid CSV data loaded.")

    # Build initial common index
    common_index: Optional[pd.DatetimeIndex] = None
    for sym_name, df_sym in raw_dfs.items():
        if df_sym.index.has_duplicates:
            logger.warning("Duplicate timestamps found in %s. Keeping first.", sym_name)
            df_sym = df_sym[~df_sym.index.duplicated(keep='first')]
            raw_dfs[sym_name] = df_sym
        common_index = df_sym.index if common_index is None else common_index.intersection(df_sym.index)

    required_len = min_common_points_param + (eval_lag if include_lag_in_required_length else 0)

    # Prune until we have enough common length
    dropped: List[str] = []

    def _common_from(dfs: Dict[str, pd.DataFrame]) -> Optional[pd.DatetimeIndex]:
        ci: Optional[pd.DatetimeIndex] = None
        for dfx in dfs.values():
            ci = dfx.index if ci is None else ci.intersection(dfx.index)
        return ci

    while True:
        current_common = _common_from(raw_dfs)
        cur_len = 0 if current_common is None else len(current_common)
        if cur_len >= required_len:
            common_index = current_common
            break
        if len(raw_dfs) <= 2:
            raise DataLoadError(
                f"Not enough common history. Need {required_len}, got {cur_len}. After pruning {len(dropped)} symbols, only {len(raw_dfs)} remain."
            )

        # Identify limiting symbols by bounds
        bounds = {s: (dfx.index[0], dfx.index[-1]) for s, dfx in raw_dfs.items()}
        latest_start_sym = max(bounds.items(), key=lambda kv: kv[1][0])[0]
        earliest_end_sym = min(bounds.items(), key=lambda kv: kv[1][1])[0]

        def _len_if_drop(sym: str) -> int:
            tmp = dict(raw_dfs)
            tmp.pop(sym, None)
            ci = _common_from(tmp)
            return 0 if ci is None else len(ci)

        if _len_if_drop(latest_start_sym) >= _len_if_drop(earliest_end_sym):
            to_drop = latest_start_sym
        else:
            to_drop = earliest_end_sym
        dropped.append(to_drop)
        raw_dfs.pop(to_drop, None)

    # If fixed-length strategies, trim to exact window length (eval points + lag)
    if strategy_param in ("common_1200", "specific_long_10k"):
        need = min_common_points_param + (eval_lag if fixed_trim_include_lag else 0)
        if len(common_index) > need:
            common_index = common_index[-need:]

    # Align all dataframes to the final index and recompute forward returns
    aligned = OrderedDict()
    for sym in sorted(raw_dfs.keys()):
        df_sym = raw_dfs[sym].reindex(common_index).ffill().bfill()
        if recompute_ret_fwd:
            try:
                df_sym["ret_fwd"] = df_sym["close"].pct_change(periods=eval_lag).shift(-eval_lag)
            except Exception:
                df_sym["ret_fwd"] = df_sym["close"].pct_change(periods=1).shift(-1)
        aligned[sym] = df_sym

    diags = DataDiagnostics(
        n_symbols_before=len(aligned) + len(dropped),
        n_symbols_after=len(aligned),
        dropped_symbols=dropped,
        overlap_len=len(common_index),
        overlap_start=common_index.min() if common_index is not None else None,
        overlap_end=common_index.max() if common_index is not None else None,
    )

    # Friendly summary for operators – concise and consistent
    try:
        kept = list(aligned.keys())
        logger.info(
            "Aligned %d→%d symbols | overlap %d (%s → %s) | dropped: %s",
            diags.n_symbols_before,
            diags.n_symbols_after,
            diags.overlap_len,
            diags.overlap_start,
            diags.overlap_end,
            ", ".join(dropped) if dropped else "none",
        )
        logger.debug("Kept symbols: %s", ", ".join(kept))
    except Exception:
        pass

    return DataBundle(aligned, common_index, list(aligned.keys()), diags)
