from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    OrderedDict as OrderedDictType,
    Tuple,
)

import numpy as np
import pandas as pd  # For DataFrame rolling in hold period

from alpha_evolve.evolution.data import DERIVED_VECTOR_FEATURES, get_sector_groups
from .metrics import compute_max_drawdown

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from alpha_evolve.programs import AlphaProgram  # Use the actual class from the framework


_FEATURE_BUNDLE_CACHE: Dict[
    Tuple[int, Tuple[Tuple[str, int], ...], Tuple[str, ...]],
    "FeatureBundle",
] = {}

_DERIVED_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "market_rel_close_t": ("closes",),
    "market_rel_ret1d_t": ("ret1d",),
    "market_zclose_t": ("closes",),
    "btc_ratio_proxy_t": ("closes",),
    "regime_volatility_t": ("vol20",),
    "regime_momentum_t": ("trend_5_20",),
    "cross_btc_momentum_t": ("trend_5_20",),
    "sector_momentum_diff_t": ("trend_5_20",),
    "market_dispersion_t": ("closes",),
}


@dataclass(slots=True)
class FeatureBundle:
    time_index: pd.DatetimeIndex
    features_per_time: List[Dict[str, Any]]

# Constants that might be needed from alpha_evolve.programs if not passed explicitly
# For now, assume SCALAR_FEATURE_NAMES and CROSS_SECTIONAL_FEATURE_VECTOR_NAMES are passed or globally available
# in the calling context (e.g. backtest_evolved_alphas.py)
# For cleaner design, these should ideally be passed to backtest_cross_sectional_alpha if they can vary.

# Helper function to scale signals
def _scale_signal_cross_sectionally(raw_signal_vector: np.ndarray, method: str, winsor_p: float | None = None) -> np.ndarray:
    if raw_signal_vector.size == 0:
        return raw_signal_vector
    
    clean_signal_vector = np.nan_to_num(raw_signal_vector, nan=0.0, posinf=0.0, neginf=0.0)

    if method == "sign":
        scaled = np.sign(clean_signal_vector)
    elif method == "rank":
        # Tie-aware average ranks mapped to [-1, 1]
        n = clean_signal_vector.size
        if n <= 1:
            scaled = np.zeros_like(clean_signal_vector)
        else:
            order = np.argsort(clean_signal_vector, kind="mergesort")
            ranks = np.empty(n, dtype=float)
            xs = clean_signal_vector[order]
            boundaries = np.empty(n + 1, dtype=bool)
            boundaries[0] = True
            if n > 1:
                boundaries[1:-1] = xs[1:] != xs[:-1]
            else:
                boundaries[1:-1] = False
            boundaries[-1] = True
            idx = np.flatnonzero(boundaries)
            for i in range(len(idx) - 1):
                start = idx[i]
                end = idx[i + 1]
                avg_rank = 0.5 * (start + end - 1)
                ranks[order[start:end]] = avg_rank
            scaled = (ranks / (n - 1 + 1e-9)) * 2.0 - 1.0
    elif method == "madz" or method == "mad":
        med = np.nanmedian(clean_signal_vector)
        mad = np.nanmedian(np.abs(clean_signal_vector - med))
        scale = 1.4826 * mad
        if scale < 1e-9:
            scaled = np.zeros_like(clean_signal_vector)
        else:
            scaled = (clean_signal_vector - med) / scale
    elif method == "winsor":
        # Use configured tail probability if provided; clamp to [0, 0.2]
        p = 0.02 if winsor_p is None else float(min(max(winsor_p, 0.0), 0.2))
        lo = np.nanquantile(clean_signal_vector, p)
        hi = np.nanquantile(clean_signal_vector, 1.0 - p)
        w = np.clip(clean_signal_vector, lo, hi)
        mu = np.nanmean(w)
        sd = np.nanstd(w)
        if sd < 1e-9:
            scaled = np.zeros_like(clean_signal_vector)
        else:
            scaled = (w - mu) / sd
    else: # Default: z-score
        mu = np.nanmean(clean_signal_vector) 
        sd = np.nanstd(clean_signal_vector)
        if sd < 1e-9 :
            scaled = np.zeros_like(clean_signal_vector)
        else:
            scaled = (clean_signal_vector - mu) / sd
    
    return np.clip(scaled, -1, 1)

def _max_drawdown(equity_curve: np.ndarray) -> float:
    """Backward-compatible wrapper around the shared helper."""

    return compute_max_drawdown(equity_curve)


def _build_feature_bundle(
    aligned_dfs: OrderedDictType[str, pd.DataFrame],
    stock_symbols: List[str],
    common_time_index: pd.DatetimeIndex,
    scalar_feature_names: List[str],
    cross_sectional_feature_vector_names: List[str],
    sector_groups_vec: np.ndarray,
) -> FeatureBundle:
    time_index = common_time_index[:-1]
    n_bars = len(time_index)
    n_stocks = len(stock_symbols)
    if n_bars == 0 or n_stocks == 0:
        return FeatureBundle(time_index=time_index, features_per_time=[])

    cross_feat_set = set(cross_sectional_feature_vector_names)
    base_columns: set[str] = set()
    for feat_name in cross_sectional_feature_vector_names:
        if feat_name == "sector_id_vector":
            continue
        if feat_name in DERIVED_VECTOR_FEATURES:
            deps = _DERIVED_DEPENDENCIES.get(feat_name, ())
            base_columns.update(deps)
        base_col = feat_name[:-2] if feat_name.endswith("_t") else feat_name
        base_columns.add(base_col)

    reindexed_frames: Dict[str, pd.DataFrame] = {
        sym: aligned_dfs[sym].reindex(time_index) for sym in stock_symbols
    }

    base_matrices: Dict[str, np.ndarray] = {}
    for col in base_columns:
        mat = np.zeros((n_bars, n_stocks), dtype=float)
        for sym_idx, sym in enumerate(stock_symbols):
            df_sym = reindexed_frames[sym]
            if col in df_sym.columns:
                values = df_sym[col].to_numpy(dtype=float, copy=False)
                mat[:, sym_idx] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        base_matrices[col] = mat

    eps = 1e-9
    derived_matrices: Dict[str, np.ndarray] = {}

    closes_mat = base_matrices.get("closes")
    if closes_mat is None:
        closes_mat = np.zeros((n_bars, n_stocks), dtype=float)

    if "market_rel_close_t" in cross_feat_set:
        mean_close = np.mean(closes_mat, axis=1, keepdims=True)
        rel_close = np.divide(
            closes_mat,
            mean_close,
            out=np.zeros_like(closes_mat),
            where=np.abs(mean_close) > eps,
        ) - 1.0
        derived_matrices["market_rel_close_t"] = np.nan_to_num(
            rel_close, nan=0.0, posinf=0.0, neginf=0.0
        )

    if "market_zclose_t" in cross_feat_set:
        mean_close = np.mean(closes_mat, axis=1, keepdims=True)
        std_close = np.std(closes_mat, axis=1, ddof=0, keepdims=True)
        zclose = np.divide(
            closes_mat - mean_close,
            std_close,
            out=np.zeros_like(closes_mat),
            where=std_close > eps,
        )
        derived_matrices["market_zclose_t"] = np.nan_to_num(
            zclose, nan=0.0, posinf=0.0, neginf=0.0
        )

    if "btc_ratio_proxy_t" in cross_feat_set:
        reference_idx = 0
        for idx, sym in enumerate(stock_symbols):
            sym_upper = sym.upper()
            if "BTC" in sym_upper or "SPY" in sym_upper or "^GSPC" in sym_upper:
                reference_idx = idx
                break
        ref_close = closes_mat[:, [reference_idx]]
        btc_ratio = np.divide(
            closes_mat,
            ref_close,
            out=np.zeros_like(closes_mat),
            where=np.abs(ref_close) > eps,
        ) - 1.0
        derived_matrices["btc_ratio_proxy_t"] = np.nan_to_num(
            btc_ratio, nan=0.0, posinf=0.0, neginf=0.0
        )

    if "market_dispersion_t" in cross_feat_set:
        median_close = np.median(closes_mat, axis=1, keepdims=True)
        dispersion = closes_mat - median_close
        derived_matrices["market_dispersion_t"] = np.nan_to_num(
            dispersion, nan=0.0, posinf=0.0, neginf=0.0
        )

    ret1d_mat = base_matrices.get("ret1d")
    if ret1d_mat is None:
        ret1d_mat = np.zeros((n_bars, n_stocks), dtype=float)

    if "market_rel_ret1d_t" in cross_feat_set:
        mean_ret = np.mean(ret1d_mat, axis=1, keepdims=True)
        rel_ret = ret1d_mat - mean_ret
        derived_matrices["market_rel_ret1d_t"] = np.nan_to_num(
            rel_ret, nan=0.0, posinf=0.0, neginf=0.0
        )

    vol20_mat = base_matrices.get("vol20")
    if vol20_mat is None:
        vol20_mat = np.zeros((n_bars, n_stocks), dtype=float)

    if "regime_volatility_t" in cross_feat_set:
        mean_vol20 = np.mean(vol20_mat, axis=1, keepdims=True)
        rel_vol = np.divide(
            vol20_mat,
            mean_vol20,
            out=np.zeros_like(vol20_mat),
            where=np.abs(mean_vol20) > eps,
        ) - 1.0
        derived_matrices["regime_volatility_t"] = np.nan_to_num(
            rel_vol, nan=0.0, posinf=0.0, neginf=0.0
        )

    trend_mat = base_matrices.get("trend_5_20")
    if trend_mat is None:
        trend_mat = np.zeros((n_bars, n_stocks), dtype=float)

    if "regime_momentum_t" in cross_feat_set:
        mean_trend = np.mean(trend_mat, axis=1, keepdims=True)
        regime_mom = trend_mat - mean_trend
        derived_matrices["regime_momentum_t"] = np.nan_to_num(
            regime_mom, nan=0.0, posinf=0.0, neginf=0.0
        )

    if "cross_btc_momentum_t" in cross_feat_set or "sector_momentum_diff_t" in cross_feat_set:
        reference_idx = 0
        for idx, sym in enumerate(stock_symbols):
            sym_upper = sym.upper()
            if "BTC" in sym_upper or "SPY" in sym_upper or "^GSPC" in sym_upper:
                reference_idx = idx
                break

    if "cross_btc_momentum_t" in cross_feat_set:
        ref_trend = trend_mat[:, [reference_idx]]
        cross_btc = trend_mat - ref_trend
        derived_matrices["cross_btc_momentum_t"] = np.nan_to_num(
            cross_btc, nan=0.0, posinf=0.0, neginf=0.0
        )

    if "sector_momentum_diff_t" in cross_feat_set:
        sector_diff = np.zeros_like(trend_mat)
        for sector_id in np.unique(sector_groups_vec):
            mask = sector_groups_vec == sector_id
            if not np.any(mask):
                continue
            sector_mean = np.mean(trend_mat[:, mask], axis=1, keepdims=True)
            sector_diff[:, mask] = trend_mat[:, mask] - sector_mean
        derived_matrices["sector_momentum_diff_t"] = np.nan_to_num(
            sector_diff, nan=0.0, posinf=0.0, neginf=0.0
        )

    sector_vec = sector_groups_vec.astype(float, copy=False)
    scalar_defaults: Dict[str, float] = {}
    for sc_name in scalar_feature_names:
        if sc_name == "const_1":
            scalar_defaults[sc_name] = 1.0
        elif sc_name == "const_neg_1":
            scalar_defaults[sc_name] = -1.0
        else:
            scalar_defaults[sc_name] = 0.0

    features_per_time: List[Dict[str, Any]] = []
    zero_template = np.zeros(n_stocks, dtype=float)

    for t in range(n_bars):
        feat_dict: Dict[str, Any] = dict(scalar_defaults)
        for feat_name in cross_sectional_feature_vector_names:
            if feat_name == "sector_id_vector":
                feat_dict[feat_name] = sector_vec
                continue
            if feat_name in derived_matrices:
                feat_dict[feat_name] = derived_matrices[feat_name][t]
                continue
            base_col = feat_name[:-2] if feat_name.endswith("_t") else feat_name
            base_mat = base_matrices.get(base_col)
            if base_mat is None:
                feat_dict[feat_name] = zero_template.copy()
            else:
                feat_dict[feat_name] = base_mat[t]
        features_per_time.append(feat_dict)

    return FeatureBundle(time_index=time_index, features_per_time=features_per_time)


def _get_feature_bundle_cached(
    aligned_dfs: OrderedDictType[str, pd.DataFrame],
    stock_symbols: List[str],
    common_time_index: pd.DatetimeIndex,
    scalar_feature_names: List[str],
    cross_sectional_feature_vector_names: List[str],
    sector_groups_vec: np.ndarray,
) -> FeatureBundle:
    cache_key = (
        id(common_time_index),
        tuple((sym, id(aligned_dfs[sym])) for sym in stock_symbols),
        tuple(cross_sectional_feature_vector_names),
        tuple(scalar_feature_names),
    )
    cached = _FEATURE_BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bundle = _build_feature_bundle(
        aligned_dfs,
        stock_symbols,
        common_time_index,
        scalar_feature_names,
        cross_sectional_feature_vector_names,
        sector_groups_vec,
    )
    _FEATURE_BUNDLE_CACHE[cache_key] = bundle
    return bundle

# Main backtesting function
def backtest_cross_sectional_alpha(
    prog: AlphaProgram,
    aligned_dfs: OrderedDictType[str, pd.DataFrame],
    common_time_index: pd.DatetimeIndex,
    stock_symbols: List[str],
    n_stocks: int,
    fee_bps: float,
    lag: int, 
    hold: int, 
    scale_method: str,
    long_short_n: int,
    initial_state_vars_config: Dict[str, str], # e.g. {"prev_s1_vec": "vector"}
    scalar_feature_names: List[str], # Pass these from calling script
    cross_sectional_feature_vector_names: List[str], # Pass these
    winsor_p: float | None = None,
    debug_prints: bool = False, # For optional debug prints
    annualization_factor: float | None = 252, # Default for daily equities; if None, infer from index
    stop_loss_pct: float = 0.0,
    # Optional risk controls
    sector_neutralize_positions: bool = False,
    volatility_target: float = 0.0,
    volatility_lookback: int = 30,
    max_leverage: float = 2.0,
    min_leverage: float = 0.25,
    dd_limit: float = 0.0,
    dd_reduction: float = 0.5,
    stress_config: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Backtest a cross‑sectional trading rule.

    Parameters
    ----------
    prog : AlphaProgram
        Program that produces a raw signal for each asset on every bar.
    aligned_dfs : OrderedDict[str, pd.DataFrame]
        Per‑symbol data frames aligned on ``common_time_index``.  Each data
        frame must provide at least ``ret_fwd`` (next bar return), ``close``,
        ``high`` and ``low`` columns.  Additional feature columns referenced by
        ``scalar_feature_names`` and ``cross_sectional_feature_vector_names``
        are expected to be present as well.
    common_time_index : pd.DatetimeIndex
        Ordered timestamps over which the backtest iterates.
    stock_symbols : list[str]
        Ordered universe of symbols matching the keys of ``aligned_dfs``.
    n_stocks : int
        Size of the universe; used for array allocations.
    fee_bps : float
        One‑way transaction cost in basis points.
    lag : int
        Execution delay in bars.  Positions are shifted by this amount.
    hold : int
        Holding period.  When greater than one, positions are averaged over
        the previous ``hold`` bars.
    scale_method : str
        Cross‑sectional scaling applied to raw signals (e.g. ``zscore``,
        ``rank``, ``sign``, ``winsor``).
    long_short_n : int
        When positive, constructs an equal‑weighted long/short basket of the
        top/bottom ``long_short_n`` names each bar.
    initial_state_vars_config : Dict[str, str]
        Mapping of state variable names to their type (``vector``, ``matrix``,
        or scalar) used to initialise ``program_state`` returned by
        ``prog.new_state``.
    scalar_feature_names : list[str]
        Names of scalar features provided to the program.
    cross_sectional_feature_vector_names : list[str]
        Names of vector features provided to the program.
    winsor_p : float, optional
        Tail probability used when ``scale_method`` is ``"winsor"``.
    debug_prints : bool, optional
        If ``True`` prints intermediate diagnostics.
    annualization_factor : float or None, optional
        Bars per year used for annualising returns.  When ``None`` the value
        is inferred from ``common_time_index`` spacing.
    stop_loss_pct : float, optional
        Intrabar percentage stop loss applied when ``lag == 1``.

    Optional Risk Controls
    ----------------------
    sector_neutralize_positions : bool
        Demeans positions within sector groups obtained from
        ``get_sector_groups``.
    volatility_target : float
        Target portfolio volatility.  Positions are scaled to reach this
        target using a ``volatility_lookback`` window and are further bounded
        by ``max_leverage``/``min_leverage``.
    dd_limit : float
        Maximum allowed drawdown.  If breached, exposure is multiplied by
        ``dd_reduction``.
    dd_reduction : float
        Exposure multiplier applied when the drawdown limit is exceeded.
    stress_config : dict, optional
        Parameters controlling stress-test metrics. When provided the
        resulting dictionary includes a ``"Stress"`` entry with performance
        statistics under heightened costs and amplified downside moves. Keys
        include ``"fee_bps"``, ``"slippage_bps"`` and ``"shock_scale"``.

    Assumptions
    -----------
    The input data frames are assumed to be free of missing timestamps and to
    contain numeric data.  Sector identifiers derived from
    ``get_sector_groups`` are expected to be non‑zero for at least one asset.

    Returns
    -------
    Dict[str, Any]
        Dictionary of performance metrics including ``Sharpe``,
        ``AnnReturn``, ``AnnVol``, ``MaxDD``, ``Turnover`` and diagnostic
        series such as ``EquityCurve`` and ``ExposureMult``.

    Side Effects
    ------------
    The function mutates ``program_state`` in place and may emit debug
    information to stdout when ``debug_prints`` is enabled.
    """
    
    program_state: Dict[str, Any] = prog.new_state()
    for s_name, s_type in initial_state_vars_config.items():
        if s_name not in program_state:
            if s_type == "vector":
                program_state[s_name] = np.zeros(n_stocks)
            elif s_type == "matrix":
                program_state[s_name] = np.zeros((n_stocks, n_stocks))
            else:
                program_state[s_name] = 0.0

    raw_signals_over_time: List[np.ndarray] = []

    # sector ids are constant across time; compute once for efficiency
    sector_groups_vec = get_sector_groups(stock_symbols).astype(float)
    sector_groups_vec_int = sector_groups_vec.astype(int)
    if np.all(sector_groups_vec == 0):
        raise RuntimeError("sector_id_vector is all zeros – check data handling")

    sector_bucket_masks: Optional[List[np.ndarray]] = None
    if sector_neutralize_positions:
        masks: List[np.ndarray] = []
        for gid in np.unique(sector_groups_vec_int):
            mask = sector_groups_vec_int == gid
            if np.any(mask):
                masks.append(mask)
        if len(masks) > 1 and not any(np.all(mask) for mask in masks):
            sector_bucket_masks = masks

    feature_bundle = _get_feature_bundle_cached(
        aligned_dfs,
        stock_symbols,
        common_time_index,
        scalar_feature_names,
        cross_sectional_feature_vector_names,
        sector_groups_vec,
    )
    eval_time_index = feature_bundle.time_index
    features_sequence = feature_bundle.features_per_time
    if len(features_sequence) != len(eval_time_index):
        raise RuntimeError("Feature bundle mismatch between features and time index")

    # For reproducibility of AlphaProgram.eval if it has stochastic elements (unlikely for these ops)
    # np.random.seed(current_seed) # Seed is usually handled at a higher level (main script)

    for timestamp, features_at_t in zip(eval_time_index, features_sequence):
        try:
            signal_vector_t = prog.eval(features_at_t, program_state, n_stocks)
            signal_vector_t = np.nan_to_num(
                signal_vector_t, nan=0.0, posinf=0.0, neginf=0.0
            )
            raw_signals_over_time.append(signal_vector_t)
        except Exception:
            raw_signals_over_time.append(np.zeros(n_stocks))

    if not raw_signals_over_time:
        return {"Sharpe": 0.0, "AnnReturn": 0.0, "AnnVol": 0.0, "MaxDD": 0.0, "Turnover": 0.0, "Bars": 0, "Error": "No signals generated"}

    signal_matrix = np.array(raw_signals_over_time)

    if debug_prints:
        logger.debug(
            f"Debug (core_logic): Raw signal_matrix σ_cross_sectional (first 5): {signal_matrix.std(axis=1)[:5]}"
        )
    if debug_prints:
        logger.debug(
            ">>> signal_matrix[:5,:5] std: %s",
            np.std(signal_matrix[:5, :], axis=1, ddof=0),
        )
    if debug_prints:
        logger.debug(">>> first few rows of signal_matrix: %s", signal_matrix[:3, :4])

    target_positions_matrix = np.zeros_like(signal_matrix)
    for t in range(signal_matrix.shape[0]):
        scaled_signal_t = _scale_signal_cross_sectionally(signal_matrix[t, :], scale_method, winsor_p)
        if long_short_n > 0:
            k = min(long_short_n, n_stocks // 2)
            order = np.argsort(scaled_signal_t)
            long_idx = order[-k:]
            short_idx = order[:k]
            ls_vector = np.zeros_like(scaled_signal_t)
            ls_vector[long_idx] = 1.0
            ls_vector[short_idx] = -1.0
            scaled_signal_t = ls_vector
        # Center cross-section and optionally sector-neutralize
        centered_signal_t = scaled_signal_t - np.mean(scaled_signal_t)
        neutralized_signal_t = centered_signal_t
        if sector_bucket_masks:
            # Demean within sector buckets using precomputed groups
            for mask in sector_bucket_masks:
                neutralized_signal_t[mask] -= np.mean(neutralized_signal_t[mask])
        # Final L1 neutralization
        sum_abs_centered_signal = np.sum(np.abs(neutralized_signal_t))
        if sum_abs_centered_signal > 1e-9:
            neutralized_signal_t = neutralized_signal_t / sum_abs_centered_signal
        else:
            neutralized_signal_t = np.zeros_like(neutralized_signal_t)
        target_positions_matrix[t, :] = neutralized_signal_t

    if debug_prints:
        logger.debug(
            f"Debug (core_logic): Target_positions_matrix σ_cross_sectional (first 5): {target_positions_matrix.std(axis=1)[:5]}"
        )

    if hold > 1:
        df_target_pos = pd.DataFrame(target_positions_matrix)
        df_held_pos = df_target_pos.rolling(window=hold, min_periods=1).mean()
        held_positions_temp = df_held_pos.values
        for t in range(held_positions_temp.shape[0]): # Re-neutralize after rolling
            current_pos_t = held_positions_temp[t,:]
            mean_pos_t = np.mean(current_pos_t)
            centered_pos_t = current_pos_t - mean_pos_t
            sum_abs_centered_pos = np.sum(np.abs(centered_pos_t))
            target_positions_matrix[t,:] = (centered_pos_t / sum_abs_centered_pos) if sum_abs_centered_pos > 1e-9 else np.zeros_like(centered_pos_t)
    
    actual_positions = np.zeros_like(target_positions_matrix)
    if lag > 0 and target_positions_matrix.shape[0] > lag:
        actual_positions[lag:, :] = target_positions_matrix[:-lag, :]
    elif lag == 0:
        actual_positions = target_positions_matrix
    
    ret_fwd_matrix = np.zeros_like(signal_matrix)
    close_t_matrix = np.zeros_like(signal_matrix)
    next_high_matrix = np.zeros_like(signal_matrix)
    next_low_matrix = np.zeros_like(signal_matrix)
    idx_for_returns = eval_time_index[: signal_matrix.shape[0]]
    for i, sym in enumerate(stock_symbols):
        ret_fwd_values = aligned_dfs[sym]["ret_fwd"].loc[idx_for_returns].values
        ret_fwd_matrix[:, i] = np.nan_to_num(ret_fwd_values, nan=0.0, posinf=0.0, neginf=0.0)
        # For simple intrabar stop-loss modeling (lag==1), we need entry close
        # at t and next bar's high/low.
        close_t_vals = aligned_dfs[sym]["close"].loc[idx_for_returns].values
        close_t_matrix[:, i] = np.nan_to_num(close_t_vals, nan=0.0, posinf=0.0, neginf=0.0)
        # Shift high/low by -1 so index t corresponds to the next bar extremes
        try:
            next_high_vals = aligned_dfs[sym]["high"].shift(-1).loc[idx_for_returns].values
            next_low_vals = aligned_dfs[sym]["low"].shift(-1).loc[idx_for_returns].values
        except Exception:
            next_high_vals = np.zeros_like(close_t_vals)
            next_low_vals = np.zeros_like(close_t_vals)
        next_high_matrix[:, i] = np.nan_to_num(next_high_vals, nan=0.0, posinf=0.0, neginf=0.0)
        next_low_matrix[:, i] = np.nan_to_num(next_low_vals, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply optional intrabar stop-loss (per-asset). Supported for lag==1.
    # If enabled and lag!=1, the stop is ignored to avoid lookahead bias.
    ret_used_matrix = ret_fwd_matrix.copy()
    # We'll compute per-bar extra stop costs after exposure scaling.
    stop_mask_matrix = np.zeros_like(ret_fwd_matrix, dtype=bool)
    stop_hits_total = 0
    stop_hit_bars = 0
    if stop_loss_pct and stop_loss_pct > 0.0 and lag == 1:
        s = float(stop_loss_pct)
        # For each time t and asset i, determine if stop is hit during next bar.
        # Long: next_low <= entry*(1-s) → realized return capped at -s
        # Short: next_high >= entry*(1+s) → realized return capped at +s
        entry_price = close_t_matrix
        long_mask = (actual_positions > 0)
        short_mask = (actual_positions < 0)
        long_stop = (next_low_matrix <= entry_price * (1.0 - s)) & long_mask
        short_stop = (next_high_matrix >= entry_price * (1.0 + s)) & short_mask
        # Cap underlying return used
        stop_mask = (long_stop | short_stop)
        stop_mask_matrix = stop_mask
        # Cap underlying return used
        ret_used_matrix = np.where(long_stop, -s, ret_used_matrix)
        ret_used_matrix = np.where(short_stop, +s, ret_used_matrix)
        # Stats (notional and extra costs computed later after scaling)
        stop_hits_total = int(np.sum(stop_mask))
        stop_hit_bars = int(np.sum(np.any(stop_mask, axis=1)))

    # Base (unit-gross) portfolio returns before costs
    base_returns = np.sum(actual_positions * ret_used_matrix, axis=1)

    # Exposure scaling via volatility targeting and drawdown limiter
    T = base_returns.shape[0]
    exposure_mult = np.ones(T, dtype=float)
    fee_rate = (fee_bps * 1e-4)
    stress_cfg = stress_config or {}
    stress_fee_bps = float(stress_cfg.get("fee_bps", fee_bps * 2.0))
    stress_slippage_bps = float(stress_cfg.get("slippage_bps", fee_bps))
    stress_shock_scale_bt = float(stress_cfg.get("shock_scale", 1.5))
    stress_tail_fee_bps = float(stress_cfg.get("tail_fee_bps", stress_fee_bps))
    stress_tail_slippage_bps = float(stress_cfg.get("tail_slippage_bps", stress_slippage_bps))
    stress_tail_shock_scale = float(stress_cfg.get("tail_shock_scale", stress_shock_scale_bt * 1.5))
    stress_fee_rate = (stress_fee_bps + stress_slippage_bps) * 1e-4
    eps = 1e-9
    # Forward compute scaled positions, costs, and returns
    scaled_positions = np.zeros_like(actual_positions)
    daily_portfolio_returns_net = np.zeros(T, dtype=float)
    abs_pos_diff_sum = np.zeros(T, dtype=float)
    equity = 1.0
    peak = 1.0
    per_bar_stop_hits = np.zeros(T, dtype=float)
    for t in range(T):
        # Volatility targeting multiplier based on base returns history
        mult = 1.0
        if volatility_target and volatility_target > 0.0:
            start = max(0, t - int(max(1, volatility_lookback)))
            if t - start >= 2:
                realized = np.std(base_returns[start:t], ddof=0)
                if realized > eps:
                    mult = float(volatility_target) / realized
            # Clamp leverage
            mult = float(np.clip(mult, min_leverage, max_leverage))
        # Drawdown limiter (based on net equity up to t-1)
        if dd_limit and dd_limit > 0.0:
            dd = (equity - peak) / (peak + eps)
            if dd < -abs(dd_limit):
                mult *= float(dd_reduction)
        exposure_mult[t] = mult
        # Apply multiplier to positions
        scaled_positions[t, :] = mult * actual_positions[t, :]
        # Transaction costs (position change costs)
        prev = scaled_positions[t - 1, :] if t > 0 else np.zeros(n_stocks)
        pos_diff_t = scaled_positions[t, :] - prev
        abs_pos_diff_sum[t] = np.sum(np.abs(pos_diff_t))
        # Extra stop costs this bar using scaled exposure (intrabar exit)
        extra_stop_cost_t = 0.0
        if stop_loss_pct and stop_loss_pct > 0.0 and lag == 1:
            per_bar_stop_notional = np.sum(np.abs(scaled_positions[t, :]) * stop_mask_matrix[t, :])
            extra_stop_cost_t = fee_rate * per_bar_stop_notional
            per_bar_stop_hits[t] = float(np.sum(stop_mask_matrix[t, :]))
        # Net return this bar
        gross_ret_t = np.dot(scaled_positions[t, :], ret_used_matrix[t, :])
        transaction_costs_t = fee_rate * abs_pos_diff_sum[t] + extra_stop_cost_t
        ret_net_t = gross_ret_t - transaction_costs_t
        daily_portfolio_returns_net[t] = ret_net_t
        # Update equity/peak for next bar drawdown logic
        equity *= (1.0 + ret_net_t)
        if equity > peak:
            peak = equity

    if not np.any(actual_positions) or not np.any(daily_portfolio_returns_net):
        return {
            "Sharpe": 0.0,
            "AnnReturn": 0.0,
            "AnnVol": 0.0,
            "MaxDD": 0.0,
            "Turnover": 0.0,
            "Bars": len(daily_portfolio_returns_net),
            "Error": "No trades executed",
        }
    
    if len(daily_portfolio_returns_net) > 0:
        mean_ret_calc = np.mean(daily_portfolio_returns_net)
        std_ret_calc = np.std(daily_portfolio_returns_net, ddof=0)
        if debug_prints:
            logger.debug(
                f"DEBUG (core_logic): PnL mean {mean_ret_calc:.6e} std {std_ret_calc:.6e}"
            )

    if len(daily_portfolio_returns_net) < 2:
        return {"Sharpe": 0.0, "AnnReturn": 0.0, "AnnVol": 0.0, "MaxDD": 0.0, "Turnover": 0.0, "Bars": len(daily_portfolio_returns_net)}

    # Determine annualization factor (bars per year)
    if annualization_factor is None:
        try:
            diffs = pd.Series(common_time_index).diff().dropna()
            if not diffs.empty:
                sec = diffs.median().total_seconds()
                if sec > 0:
                    if abs(sec - 86400.0) <= 1200.0:  # ≈ one trading day
                        annualization_factor = 252.0
                    elif abs(sec - 14400.0) <= 600.0:  # ≈ 4h bars
                        annualization_factor = float(365 * 6)
                    else:
                        annualization_factor = float((365.0 * 24.0 * 3600.0) / sec)
                else:
                    annualization_factor = 252.0
            else:
                annualization_factor = 252.0
        except Exception:
            annualization_factor = 252.0
    else:
        annualization_factor = float(annualization_factor)

    equity_curve = np.cumprod(1 + daily_portfolio_returns_net)
    peak_curve = np.maximum.accumulate(equity_curve)
    drawdown_series = (equity_curve - peak_curve) / (peak_curve + 1e-9)
    mean_ret = np.mean(daily_portfolio_returns_net)
    std_ret = np.std(daily_portfolio_returns_net, ddof=0)

    sharpe_ratio = (mean_ret / (std_ret + 1e-9)) * np.sqrt(annualization_factor)
    total_return_val = equity_curve[-1] - 1.0
    num_years = len(daily_portfolio_returns_net) / annualization_factor
    annualized_return = ((1.0 + total_return_val) ** (1.0 / num_years)) - 1.0 if num_years > 0 else 0.0
    annualized_volatility = std_ret * np.sqrt(annualization_factor)
    max_dd = _max_drawdown(equity_curve)
    avg_daily_turnover_fraction = np.mean(abs_pos_diff_sum) / 2.0

    result = {
        "Sharpe": sharpe_ratio,
        "AnnReturn": annualized_return,
        "AnnVol": annualized_volatility,
        "MaxDD": max_dd,
        "Turnover": avg_daily_turnover_fraction,
        "Bars": len(daily_portfolio_returns_net),
        "StopHits": stop_hits_total,
        "StopBars": stop_hit_bars,
        "VolTarget": float(volatility_target),
        "DDLimit": float(dd_limit),
        # Time series for diagnostics
        "EquityCurve": equity_curve,
        "ExposureMult": exposure_mult,
        "Drawdown": drawdown_series,
        "StopHitsPerBar": per_bar_stop_hits,
        "RetNet": daily_portfolio_returns_net,
    }

    scenario_specs = {
        "base": {
            "fee_bps": stress_fee_bps,
            "slippage_bps": stress_slippage_bps,
            "shock_scale": stress_shock_scale_bt,
        }
    }
    if stress_tail_fee_bps > 0.0 or stress_tail_slippage_bps > 0.0 or stress_tail_shock_scale > 0.0:
        scenario_specs["tail"] = {
            "fee_bps": stress_tail_fee_bps,
            "slippage_bps": stress_tail_slippage_bps,
            "shock_scale": stress_tail_shock_scale if stress_tail_shock_scale > 0.0 else stress_shock_scale_bt,
        }

    stress_results: Dict[str, Dict[str, float]] = {}

    def _compute_scenario(label: str, fee_bps_local: float, slip_bps_local: float, shock_scale_local: float) -> Dict[str, float]:
        if len(daily_portfolio_returns_net) == 0:
            return {
                "Sharpe": 0.0,
                "AnnReturn": 0.0,
                "AnnVol": 0.0,
                "MaxDD": 0.0,
                "MeanRet": 0.0,
                "CostExtra": 0.0,
                "ShockScale": float(shock_scale_local),
            }
        fee_rate_local = (float(fee_bps_local) + float(slip_bps_local)) * 1e-4
        extra_costs_local = fee_rate_local * abs_pos_diff_sum
        stressed_returns_local = daily_portfolio_returns_net - extra_costs_local
        stressed_returns_local = stressed_returns_local.copy()
        scale_local = float(max(1.0, shock_scale_local))
        if stressed_returns_local.size > 0 and scale_local > 1.0:
            neg_mask = stressed_returns_local < 0.0
            pos_mask = stressed_returns_local > 0.0
            stressed_returns_local[neg_mask] *= scale_local
            stressed_returns_local[pos_mask] /= scale_local
        if stressed_returns_local.size == 0:
            return {
                "Sharpe": 0.0,
                "AnnReturn": 0.0,
                "AnnVol": 0.0,
                "MaxDD": 0.0,
                "MeanRet": 0.0,
                "CostExtra": 0.0,
                "ShockScale": float(scale_local),
            }
        stress_equity_curve_local = np.cumprod(1 + stressed_returns_local)
        stress_dd_local = _max_drawdown(stress_equity_curve_local)
        stress_mean_local = float(np.mean(stressed_returns_local))
        stress_std_local = float(np.std(stressed_returns_local, ddof=0))
        stress_sharpe_local = (stress_mean_local / (stress_std_local + 1e-9)) * np.sqrt(annualization_factor)
        if num_years > 0 and stress_equity_curve_local[-1] > 0:
            stress_ann_return_local = (stress_equity_curve_local[-1] ** (1.0 / num_years)) - 1.0
        else:
            stress_ann_return_local = 0.0
        return {
            "Sharpe": stress_sharpe_local,
            "AnnReturn": stress_ann_return_local,
            "AnnVol": stress_std_local * np.sqrt(annualization_factor),
            "MaxDD": stress_dd_local,
            "MeanRet": stress_mean_local,
            "CostExtra": float(np.mean(extra_costs_local)) if extra_costs_local.size else 0.0,
            "ShockScale": float(scale_local),
        }

    for label, spec in scenario_specs.items():
        stress_results[label] = _compute_scenario(
            label,
            spec.get("fee_bps", 0.0),
            spec.get("slippage_bps", 0.0),
            spec.get("shock_scale", 1.0),
        )

    result["Stress"] = stress_results.get("base", {
        "Sharpe": 0.0,
        "AnnReturn": 0.0,
        "AnnVol": 0.0,
        "MaxDD": 0.0,
        "MeanRet": 0.0,
        "CostExtra": 0.0,
        "ShockScale": stress_shock_scale_bt,
    })
    result["StressScenarios"] = stress_results
    baseline_tc = fee_rate * abs_pos_diff_sum if len(daily_portfolio_returns_net) > 0 else np.zeros(0)
    result["TransactionCosts"] = {
        "baseline_cost": float(np.mean(baseline_tc)) if baseline_tc.size else 0.0,
        "turnover_mean": float(np.mean(abs_pos_diff_sum)) if len(abs_pos_diff_sum) else 0.0,
    }

    return result
