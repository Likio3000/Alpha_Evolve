from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Set
from collections import OrderedDict
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from alpha_framework.alpha_framework_program import AlphaProgram # Changed from alpha_program_core
    from alpha_framework.alpha_framework_op import Op # For _uses_feature_vector
    from evolution_components import data_handling # To access data
    from evolution_components import hall_of_fame_manager as hof_manager # To get HOF penalty

from alpha_framework.alpha_framework_types import (  # Ensure these are correct based on where AlphaProgram is defined
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    FINAL_PREDICTION_VECTOR_NAME,
)


# Module-level cache (least-recently used)
@dataclass
class EvalResult:
    fitness: float
    mean_ic: float
    sharpe_proxy: float
    parsimony_penalty: float
    correlation_penalty: float
    processed_predictions: Optional[np.ndarray]
    ic_std: float = 0.0
    turnover_proxy: float = 0.0
    # Optional: fixed-weight fitness for comparability across gens (no ramping)
    fitness_static: Optional[float] = None


_eval_cache: "OrderedDict[str, EvalResult]" = OrderedDict()
_EVAL_CACHE_MAX_SIZE = 128


def _cache_set(fp: str, value: EvalResult) -> None:
    if fp in _eval_cache:
        _eval_cache.move_to_end(fp)
    elif len(_eval_cache) >= _EVAL_CACHE_MAX_SIZE:
        _eval_cache.popitem(last=False)
    _eval_cache[fp] = value

# Evaluation constants (to be passed or configured)
# These were global in evolve_alphas.py
# They should be passed into evaluate_program or configured via an init function for this module.
# For now, let's define them here with defaults that can be overridden.
_EVAL_CONFIG = {
    "parsimony_penalty_factor": 0.002,
    "max_ops_for_parsimony": 32,
    "xs_flatness_guard_threshold": 5e-3,
    "temporal_flatness_guard_threshold": 5e-3, # Renamed from flat_signal_threshold
    "early_abort_bars": 20,
    "early_abort_xs_threshold": 5e-2,
    "early_abort_t_threshold": 5e-2,
    "flat_bar_threshold": 0.25,
    "ic_scale_method": "zscore", # from args.scale
    "winsor_p": 0.01,            # winsorization tail prob for 'winsor'
    "sector_neutralize": False,  # optionally demean by sector before IC
    "sharpe_proxy_weight": 0.0,
    "ic_std_penalty_weight": 0.0,
    "turnover_penalty_weight": 0.0,
    # Fixed weights for comparability (no ramping)
    "fixed_sharpe_proxy_weight": 0.0,
    "fixed_ic_std_penalty_weight": 0.0,
    "fixed_turnover_penalty_weight": 0.0,
    "fixed_corr_penalty_weight": 0.0,
    "use_train_val_splits": False,
    "train_points": 0,
    "val_points": 0,
    # EVAL_LAG is handled by data_handling module ensuring data is sliced appropriately
}

# Lightweight, per-process evaluation stats for visibility during evolution
_EVAL_STATS = {
    "cache_hits": 0,
    "cache_misses": 0,
    "rejected_no_feature_vec": 0,
    "rejected_nan_or_inf": 0,
    "rejected_all_zero": 0,
    "early_abort_xs": 0,
    "early_abort_t": 0,
    "early_abort_flatbar": 0,
}

def reset_eval_stats() -> None:
    for k in _EVAL_STATS:
        _EVAL_STATS[k] = 0

def get_eval_stats() -> dict:
    return dict(_EVAL_STATS)

# Lightweight per-program event samples for diagnostics per generation
_EVAL_EVENTS: list = []
_EVAL_EVENTS_MAX = 200

def reset_eval_events() -> None:
    global _EVAL_EVENTS
    _EVAL_EVENTS = []

def get_eval_events() -> list:
    return list(_EVAL_EVENTS)

def _push_event(ev: dict) -> None:
    try:
        if len(_EVAL_EVENTS) < _EVAL_EVENTS_MAX:
            _EVAL_EVENTS.append(ev)
    except Exception:
        pass

def configure_evaluation(
    parsimony_penalty: float,
    max_ops: int,
    xs_flatness_guard: float,
    temporal_flatness_guard: float,
    early_abort_bars: int,
    early_abort_xs: float,
    early_abort_t: float,
    flat_bar_threshold: float,
    scale_method: str,
    sharpe_proxy_weight: float = 0.0,
    ic_std_penalty_weight: float = 0.0,
    turnover_penalty_weight: float = 0.0,
    use_train_val_splits: bool = False,
    train_points: int = 0,
    val_points: int = 0,
    *,
    sector_neutralize: bool = False,
    winsor_p: float = 0.01,
    # Optional: add deterministic jitter to parsimony penalty to improve ops variance
    parsimony_jitter_pct: float = 0.0,
    # Optional: provide fixed (non-ramped) weights for logging/secondary fitness
    fixed_sharpe_proxy_weight: Optional[float] = None,
    fixed_ic_std_penalty_weight: Optional[float] = None,
    fixed_turnover_penalty_weight: Optional[float] = None,
    fixed_corr_penalty_weight: Optional[float] = None,
    ):
    global _EVAL_CONFIG
    _EVAL_CONFIG["parsimony_penalty_factor"] = parsimony_penalty
    _EVAL_CONFIG["max_ops_for_parsimony"] = max_ops
    _EVAL_CONFIG["xs_flatness_guard_threshold"] = xs_flatness_guard
    _EVAL_CONFIG["temporal_flatness_guard_threshold"] = temporal_flatness_guard
    _EVAL_CONFIG["early_abort_bars"] = early_abort_bars
    _EVAL_CONFIG["early_abort_xs_threshold"] = early_abort_xs
    _EVAL_CONFIG["early_abort_t_threshold"] = early_abort_t
    _EVAL_CONFIG["flat_bar_threshold"] = flat_bar_threshold
    _EVAL_CONFIG["ic_scale_method"] = scale_method
    _EVAL_CONFIG["sector_neutralize"] = bool(sector_neutralize)
    _EVAL_CONFIG["winsor_p"] = float(winsor_p)
    _EVAL_CONFIG["sharpe_proxy_weight"] = sharpe_proxy_weight
    _EVAL_CONFIG["ic_std_penalty_weight"] = ic_std_penalty_weight
    _EVAL_CONFIG["turnover_penalty_weight"] = turnover_penalty_weight
    _EVAL_CONFIG["use_train_val_splits"] = use_train_val_splits
    _EVAL_CONFIG["train_points"] = int(train_points)
    _EVAL_CONFIG["val_points"] = int(val_points)
    # Fixed weights default to current (possibly ramped) values if not provided
    _EVAL_CONFIG["fixed_sharpe_proxy_weight"] = (
        float(sharpe_proxy_weight) if fixed_sharpe_proxy_weight is None else float(fixed_sharpe_proxy_weight)
    )
    _EVAL_CONFIG["fixed_ic_std_penalty_weight"] = (
        float(ic_std_penalty_weight) if fixed_ic_std_penalty_weight is None else float(fixed_ic_std_penalty_weight)
    )
    _EVAL_CONFIG["fixed_turnover_penalty_weight"] = (
        float(turnover_penalty_weight) if fixed_turnover_penalty_weight is None else float(fixed_turnover_penalty_weight)
    )
    _EVAL_CONFIG["fixed_corr_penalty_weight"] = (
        float(_EVAL_CONFIG.get("fixed_corr_penalty_weight", 0.0)) if fixed_corr_penalty_weight is None else float(fixed_corr_penalty_weight)
    )
    # Clamp jitter into [0, 1] and store
    try:
        pj = float(parsimony_jitter_pct)
    except Exception:
        pj = 0.0
    _EVAL_CONFIG["parsimony_jitter_pct"] = max(0.0, min(1.0, pj))
    logging.getLogger(__name__).debug(
        "Evaluation configured: scale=%s parsimony=%s sharpe_w=%s ic_std_w=%s turnover_w=%s splits=%s train=%s val=%s sector_neutralize=%s winsor_p=%.3f jitter=%.3f",
        scale_method,
        parsimony_penalty,
        sharpe_proxy_weight,
        ic_std_penalty_weight,
        turnover_penalty_weight,
        use_train_val_splits,
        train_points,
        val_points,
        sector_neutralize,
        winsor_p,
        _EVAL_CONFIG["parsimony_jitter_pct"],
    )


def initialize_evaluation_cache(max_size: int = 128):
    global _eval_cache, _EVAL_CACHE_MAX_SIZE
    _EVAL_CACHE_MAX_SIZE = max_size
    _eval_cache = OrderedDict()
    logging.getLogger(__name__).debug("Evaluation cache cleared and initialized.")

def _safe_corr_eval(a: np.ndarray, b: np.ndarray) -> float:  # Specific to evaluation logic's needs
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0
    # Check for near-constant arrays robustly
    std_a = np.std(a, ddof=0)
    std_b = np.std(b, ddof=0)
    if std_a < 1e-9 or std_b < 1e-9:
        return 0.0
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    
    with np.errstate(invalid='ignore'): # Suppress "invalid value encountered in true_divide"
        corr_matrix = np.corrcoef(a, b)
    
    if np.isnan(corr_matrix[0, 1]):
        # This can happen if, despite std > 1e-9, one array is effectively constant for corrcoef
        return 0.0
    return float(corr_matrix[0, 1])


def _uses_feature_vector_check(prog: AlphaProgram) -> bool:
    # This utility checks if the program's final output depends on any feature vector.
    # It uses AlphaProgram's structure (setup, predict_ops, update_ops)
    # and type definitions like CROSS_SECTIONAL_FEATURE_VECTOR_NAMES.
    
    # Re-implemented based on the original logic in evolve_alphas
    all_ops_map: Dict[str, Op] = {} # Op needs to be imported from alpha_framework.alpha_framework_op
    for op_instance in prog.setup + prog.predict_ops + prog.update_ops:
        all_ops_map[op_instance.out] = op_instance

    # Initial state vars and feature vars are not directly part of prog's ops but act as inputs.
    # We need to know which INITIAL_STATE_VARS and FEATURE_VARS are vectors.
    # This check is primarily for CROSS_SECTIONAL_FEATURE_VECTOR_NAMES.
    
    # If the final prediction name itself is a feature vector, then true.
    if FINAL_PREDICTION_VECTOR_NAME in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
        return True
        
    # If the final prediction is not defined by an op (e.g. it's a direct feature or state var)
    if FINAL_PREDICTION_VECTOR_NAME not in all_ops_map:
        # This case implies FINAL_PREDICTION_VECTOR_NAME must be an input feature/state.
        # The original check was:
        # if FINAL_PREDICTION_VECTOR_NAME not in all_ops_map and \
        #    FINAL_PREDICTION_VECTOR_NAME not in INITIAL_STATE_VARS and \
        #    FINAL_PREDICTION_VECTOR_NAME not in FEATURE_VARS:
        #     return FINAL_PREDICTION_VECTOR_NAME in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES
        # This seems a bit convoluted. If it's not an op output, it must be an input.
        # We are interested if *any* feature vector contributes.
        pass # Handled by trace below

    q: List[str] = [FINAL_PREDICTION_VECTOR_NAME]
    visited_vars: Set[str] = set()

    while q:
        current_var_name = q.pop(0) # BFS style
        if current_var_name in visited_vars:
            continue
        visited_vars.add(current_var_name)

        if current_var_name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
            return True # Found a dependency on a feature vector

        # If current_var_name is an initial state var that might be a vector,
        # and it's not further defined by an op, we don't trace it further from here (unless it's the target).
        # We are looking for *feature* vectors.

        defining_op = all_ops_map.get(current_var_name)
        if defining_op:
            for input_var_name in defining_op.inputs:
                if input_var_name not in visited_vars:
                    q.append(input_var_name)
    return False


def _scale_signal_for_ic(raw_signal_vector: np.ndarray, method: str) -> np.ndarray:
    # From evolve_alphas _scale_signal_cross_sectionally_for_ic
    if raw_signal_vector.size == 0:
        return raw_signal_vector
    
    clean_signal_vector = np.nan_to_num(raw_signal_vector, nan=0.0, posinf=0.0, neginf=0.0)

    if method == "sign":
        scaled = np.sign(clean_signal_vector)
    elif method == "rank":
        if clean_signal_vector.size <= 1:
            scaled = np.zeros_like(clean_signal_vector)
        else:
            temp = clean_signal_vector.argsort()
            ranks = np.empty_like(temp, dtype=float)
            ranks[temp] = np.arange(len(clean_signal_vector))
            scaled = (ranks / (len(clean_signal_vector) - 1 + 1e-9)) * 2.0 - 1.0
    elif method == "madz" or method == "mad":
        med = np.nanmedian(clean_signal_vector)
        mad = np.nanmedian(np.abs(clean_signal_vector - med))
        scale = 1.4826 * mad
        if scale < 1e-9:
            scaled = np.zeros_like(clean_signal_vector)
        else:
            scaled = (clean_signal_vector - med) / scale
    elif method == "winsor":
        p = float(_EVAL_CONFIG.get("winsor_p", 0.01))
        p = min(max(p, 0.0), 0.2)
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
    
    # Centering after scaling for IC calculation and HOF comparison
    # Ensure it's always centered for IC, regardless of method, this was specific to the original.
    centered_scaled = scaled - np.mean(scaled) 
    return np.clip(centered_scaled, -1, 1)

def _demean_by_groups(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Demean vector x within integer group IDs, then re-center and clip.

    Groups with ID < 0 are treated as a separate bucket. If a group
    has size 0 (shouldn't happen) it is skipped. Returns a vector of
    same shape.
    """
    if x.size == 0:
        return x
    g = groups.astype(int, copy=False)
    out = x.copy()
    # unique groups
    uniq = np.unique(g)
    for gid in uniq:
        mask = g == gid
        if not np.any(mask):
            continue
        out[mask] = out[mask] - np.mean(out[mask])
    # Re-center overall and clip to keep parity with scaling
    out = out - np.mean(out)
    return np.clip(out, -1, 1)


def _average_rank_ties(x: np.ndarray) -> np.ndarray:
    """Compute average ranks [0..n-1] with tie handling.

    Uses stable sort and assigns each equal-value run the mean of its index range.
    """
    n = x.size
    if n == 0:
        return x.astype(float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    xs = x[order]
    # boundaries mask has True at run starts and at the sentinel end
    boundaries = np.empty(n + 1, dtype=bool)
    boundaries[0] = True
    if n > 1:
        boundaries[1:-1] = xs[1:] != xs[:-1]
    else:
        boundaries[1:-1] = False
    boundaries[-1] = True
    idx = np.flatnonzero(boundaries)
    for i in range(len(idx) - 1):
        start, end = idx[i], idx[i + 1]
        avg = (start + end - 1) / 2.0
        ranks[order[start:end]] = avg
    return ranks


def evaluate_program(
    prog: AlphaProgram,
    dh_module: data_handling,  # Pass the data_handling module for access
    hof_module: hof_manager,   # Pass the hall_of_fame_manager module
    initial_prog_state_vars_config: Dict[str, Any],  # e.g. evolve_alphas.INITIAL_STATE_VARS
    return_preds: bool = True,
    ctx=None,
) -> EvalResult:
    # Uses _EVAL_CONFIG for various thresholds and penalties

    logger = logging.getLogger(__name__)

    fp = prog.fingerprint
    if fp in _eval_cache:
        _eval_cache.move_to_end(fp)
        cached = _eval_cache[fp]
        _EVAL_STATS["cache_hits"] += 1
        logger.debug("Cache hit for %s with fitness %.6f", fp, cached.fitness)
        return cached
    else:
        _EVAL_STATS["cache_misses"] += 1

    if not _uses_feature_vector_check(prog):
        logger.debug("Program %s does not use any feature vector", fp)
        _EVAL_STATS["rejected_no_feature_vec"] += 1
        _push_event({"fp": fp, "event": "no_feature_vector"})
        result = EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None)
        _cache_set(fp, result)
        return result

    # Get data, either from provided context or from the data_handling module
    if ctx is not None:
        aligned_dfs = ctx.bundle.aligned_dfs
        common_time_index = ctx.bundle.common_index
        stock_symbols = ctx.bundle.symbols
        n_stocks = len(stock_symbols)
        eval_lag = ctx.eval_lag
        sector_groups_vec = ctx.sector_ids.astype(float)
    else:
        aligned_dfs = dh_module.get_aligned_dfs()
        common_time_index = dh_module.get_common_time_index()
        stock_symbols = dh_module.get_stock_symbols()
        n_stocks = dh_module.get_n_stocks()
        eval_lag = dh_module.get_eval_lag()  # Get eval_lag from data_handling
        sector_groups_vec = dh_module.get_sector_groups(stock_symbols).astype(float)

    program_state: Dict[str, Any] = prog.new_state() # AlphaProgram's own new_state
    for s_name, s_type in initial_prog_state_vars_config.items():  # Use passed config
        if s_name not in program_state:  # Allow AlphaProgram's new_state to pre-populate
            if s_type == "vector":
                program_state[s_name] = np.zeros(n_stocks)
            elif s_type == "matrix":
                program_state[s_name] = np.zeros((n_stocks, n_stocks))  # Assuming square for now
            else:
                program_state[s_name] = 0.0


    all_raw_predictions_timeseries: List[np.ndarray] = []
    all_processed_predictions_timeseries: List[np.ndarray] = []
    daily_ic_values: List[float] = []
    pnl_sum = 0.0
    pnl_sq_sum = 0.0

    # The common_time_index from data_handling is already sliced appropriately for eval_lag
    # Loop until common_time_index.size - eval_lag -1 (for python 0-indexing)
    # Or more simply, loop `len(common_time_index) - eval_lag` times.
    # If common_time_index has N points, we can make N - eval_lag predictions.
    # Prediction at t_idx uses features at t_idx, compares with returns at t_idx + eval_lag.
    
    num_evaluation_steps = len(common_time_index) - eval_lag
    if num_evaluation_steps <= 0: # Not enough data for even one evaluation
        _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None))
        return _eval_cache[fp]

    for t_idx in range(num_evaluation_steps):
        timestamp = common_time_index[t_idx]

        if ctx is None and hasattr(dh_module, "get_features_cached"):
            features_at_t = dh_module.get_features_cached(timestamp)
        else:
            features_at_t = dh_module.get_features_at_time(
                timestamp, aligned_dfs, stock_symbols, sector_groups_vec
            )

        try:
            # prog.eval uses n_stocks for internal vector shaping/broadcasting.
            raw_predictions_t = prog.eval(features_at_t, program_state, n_stocks)

            if np.any(np.isnan(raw_predictions_t)) or np.any(np.isinf(raw_predictions_t)):
                # print(f"Program {fp} yielded NaN/Inf at t_idx {t_idx}. Aborting eval for this prog.")
                _EVAL_STATS["rejected_nan_or_inf"] += 1
                _push_event({"fp": fp, "event": "nan_or_inf"})
                _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None))
                return _eval_cache[fp]
            
            all_raw_predictions_timeseries.append(raw_predictions_t.copy())
            
            processed_predictions_t = _scale_signal_for_ic(
                raw_predictions_t, _EVAL_CONFIG["ic_scale_method"]
            )
            if _EVAL_CONFIG.get("sector_neutralize", False):
                processed_predictions_t = _demean_by_groups(processed_predictions_t, sector_groups_vec)
            all_processed_predictions_timeseries.append(processed_predictions_t)

            # Early abort checks (on raw predictions)
            if len(all_raw_predictions_timeseries) == _EVAL_CONFIG["early_abort_bars"]:
                partial_raw_preds_matrix = np.array(all_raw_predictions_timeseries)
                mean_xs_std_partial = 0.0
                flat_fraction = 0.0
                cross_sectional_stds = np.array([])
                if partial_raw_preds_matrix.ndim == 2 and partial_raw_preds_matrix.shape[1] > 0:
                    cross_sectional_stds = partial_raw_preds_matrix.std(axis=1, ddof=0)
                    mean_xs_std_partial = np.mean(cross_sectional_stds)
                    if cross_sectional_stds.size > 0:
                        flat_fraction = np.mean(cross_sectional_stds < 1e-4)
                
                mean_t_std_partial = 0.0
                if partial_raw_preds_matrix.ndim == 2 and partial_raw_preds_matrix.shape[0] > 1: # If matrix
                    mean_t_std_partial = np.mean(partial_raw_preds_matrix.std(axis=0, ddof=0))
                elif partial_raw_preds_matrix.ndim == 1 and partial_raw_preds_matrix.shape[0] > 1: # If effectively a single series over time (e.g. n_stocks=1)
                     mean_t_std_partial = partial_raw_preds_matrix.std(ddof=0)


                early_abort = False
                if mean_xs_std_partial < _EVAL_CONFIG["early_abort_xs_threshold"]:
                    logger.debug(
                        "Early abort for %s due to XS flatness %.6f < %.6f",
                        fp,
                        mean_xs_std_partial,
                        _EVAL_CONFIG["early_abort_xs_threshold"],
                    )
                    _EVAL_STATS["early_abort_xs"] += 1
                    early_abort = True
                if flat_fraction > _EVAL_CONFIG["flat_bar_threshold"]:
                    logger.debug(
                        "Early abort for %s due to flat bar fraction %.6f > %.6f",
                        fp,
                        flat_fraction,
                        _EVAL_CONFIG["flat_bar_threshold"],
                    )
                    _EVAL_STATS["early_abort_flatbar"] += 1
                    early_abort = True
                if mean_t_std_partial < _EVAL_CONFIG["early_abort_t_threshold"]:
                    logger.debug(
                        "Early abort for %s due to temporal flatness %.6f < %.6f",
                        fp,
                        mean_t_std_partial,
                        _EVAL_CONFIG["early_abort_t_threshold"],
                    )
                    _EVAL_STATS["early_abort_t"] += 1
                    early_abort = True
                if early_abort:
                    _push_event({
                        "fp": fp,
                        "event": "early_abort",
                        "mean_xs_std": float(mean_xs_std_partial),
                        "flat_fraction": float(flat_fraction),
                        "mean_t_std": float(mean_t_std_partial),
                    })
                    logger.debug(
                        "Early abort stats %s | mean_xs_std_partial %.6f (< %.6f) flat_fraction %.6f (> %.6f) mean_t_std_partial %.6f (< %.6f)",
                        fp,
                        mean_xs_std_partial,
                        _EVAL_CONFIG["early_abort_xs_threshold"],
                        flat_fraction,
                        _EVAL_CONFIG["flat_bar_threshold"],
                        mean_t_std_partial,
                        _EVAL_CONFIG["early_abort_t_threshold"],
                    )
                    result = EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None)
                    _cache_set(fp, result)
                    return result

            # IC calculation
            return_timestamp_for_ic = common_time_index[t_idx + eval_lag]
            actual_returns_t_slice = np.array([aligned_dfs[sym].loc[return_timestamp_for_ic, "ret_fwd"] for sym in stock_symbols], dtype=float)
            actual_returns_t = np.nan_to_num(actual_returns_t_slice, nan=0.0, posinf=0.0, neginf=0.0) # Clean returns

            if np.all(np.isnan(actual_returns_t)): # If all returns are NaN (e.g. end of data for some reason)
                daily_ic_values.append(0.0)
                daily_pnl = 0.0
            else:
                # rank both sides for Spearman IC (tie-aware)
                r_pred = _average_rank_ties(processed_predictions_t)
                r_rets = _average_rank_ties(actual_returns_t)
                ic_t = _safe_corr_eval(r_pred, r_rets)
                daily_ic_values.append(0.0 if np.isnan(ic_t) else ic_t)
                daily_pnl = float(np.mean(processed_predictions_t * actual_returns_t))

            pnl_sum += daily_pnl
            pnl_sq_sum += daily_pnl ** 2

        except Exception: # Broad exception during program's .eval() or IC calculation
            _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None))
            return _eval_cache[fp]

    if not daily_ic_values or not all_raw_predictions_timeseries or not all_processed_predictions_timeseries:
        _push_event({"fp": fp, "event": "exception"})
        _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None))
        return _eval_cache[fp]

    # Compute metrics (optionally on train/val splits)
    full_raw_predictions_matrix = np.array(all_raw_predictions_timeseries)
    full_processed_predictions_matrix = np.array(all_processed_predictions_timeseries)

    def _neutralize(mat: np.ndarray) -> np.ndarray:
        if mat.ndim != 2 or mat.shape[1] == 0:
            return mat
        centered = mat - mat.mean(axis=1, keepdims=True)
        l1 = np.sum(np.abs(centered), axis=1, keepdims=True)
        # Use np.divide with a mask to avoid RuntimeWarning on 0/0
        out = np.zeros_like(centered)
        np.divide(centered, l1, out=out, where=l1 > 1e-9)
        return out

    total_steps = len(daily_ic_values)
    use_splits = bool(_EVAL_CONFIG.get("use_train_val_splits", False))
    t_points = int(_EVAL_CONFIG.get("train_points", 0))
    v_points = int(_EVAL_CONFIG.get("val_points", 0))

    mean_daily_ic = float(np.mean(daily_ic_values))
    mean_pnl = pnl_sum / num_evaluation_steps
    var_pnl = pnl_sq_sum / num_evaluation_steps - mean_pnl ** 2
    std_pnl = var_pnl ** 0.5 if var_pnl > 0 else 0.0
    sharpe_proxy = mean_pnl / std_pnl if std_pnl > 1e-9 else 0.0
    ic_std = float(np.std(daily_ic_values, ddof=0)) if daily_ic_values else 0.0
    turnover_proxy = 0.0

    if use_splits and t_points > 0 and v_points > 0 and (t_points + v_points) <= total_steps:
        tr = slice(0, t_points)
        va = slice(t_points, t_points + v_points)
        ic_tr = float(np.mean(daily_ic_values[tr])) if t_points > 0 else 0.0
        ic_va = float(np.mean(daily_ic_values[va])) if v_points > 0 else 0.0
        ic_std_tr = float(np.std(daily_ic_values[tr], ddof=0)) if t_points > 1 else 0.0
        ic_std_va = float(np.std(daily_ic_values[va], ddof=0)) if v_points > 1 else 0.0

        pos_tr = _neutralize(full_processed_predictions_matrix[tr])
        pos_va = _neutralize(full_processed_predictions_matrix[va])
        turn_tr = float(np.mean(np.sum(np.abs(np.diff(pos_tr, axis=0)), axis=1)/2.0)) if pos_tr.shape[0] > 1 else 0.0
        turn_va = float(np.mean(np.sum(np.abs(np.diff(pos_va, axis=0)), axis=1)/2.0)) if pos_va.shape[0] > 1 else 0.0

        mean_daily_ic = 0.5 * (ic_tr + ic_va)
        ic_std = 0.5 * (ic_std_tr + ic_std_va)
        turnover_proxy = 0.5 * (turn_tr + turn_va)
        processed_for_hof = full_processed_predictions_matrix[va]
    else:
        pos_full = _neutralize(full_processed_predictions_matrix)
        turnover_proxy = float(np.mean(np.sum(np.abs(np.diff(pos_full, axis=0)), axis=1)/2.0)) if pos_full.shape[0] > 1 else 0.0
        processed_for_hof = full_processed_predictions_matrix

    # Linear parsimony penalty to match expected tests: factor * (size / max_ops)
    try:
        max_ops_norm = float(max(1, _EVAL_CONFIG["max_ops_for_parsimony"]))
    except Exception:
        max_ops_norm = float(max(1, prog.size))
    parsimony_norm = float(prog.size) / max_ops_norm
    parsimony_penalty = _EVAL_CONFIG["parsimony_penalty_factor"] * parsimony_norm
    # Apply deterministic jitter to parsimony penalty based on program fingerprint
    try:
        pj = float(_EVAL_CONFIG.get("parsimony_jitter_pct", 0.0))
    except Exception:
        pj = 0.0
    if pj > 1e-12:
        import hashlib
        # Map fingerprint to a stable float in [0,1)
        h = hashlib.sha1(fp.encode("utf-8")).digest()
        # Use first 8 bytes as unsigned integer for reproducible fraction
        u = int.from_bytes(h[:8], byteorder="big", signed=False) / float(2**64)
        # Scale to [-pj, +pj]
        jitter_factor = 1.0 + pj * (2.0 * u - 1.0)
        parsimony_penalty *= jitter_factor
    score = (
        mean_daily_ic
        + _EVAL_CONFIG.get("sharpe_proxy_weight", 0.0) * sharpe_proxy
        - _EVAL_CONFIG.get("ic_std_penalty_weight", 0.0) * ic_std
        - _EVAL_CONFIG.get("turnover_penalty_weight", 0.0) * turnover_proxy
        - parsimony_penalty
    )
    correlation_penalty = 0.0

    if np.all(processed_for_hof == 0):
        logger.debug("All-zero predictions – rejected")
        _EVAL_STATS["rejected_all_zero"] += 1
        score = -float('inf')

    # Flatness guards (on raw predictions)
    if full_raw_predictions_matrix.ndim == 2 and full_raw_predictions_matrix.shape[1] > 0: # XS check
        cross_sectional_stds = full_raw_predictions_matrix.std(axis=1, ddof=0)
        mean_xs_std = np.mean(cross_sectional_stds)
        if mean_xs_std < _EVAL_CONFIG["xs_flatness_guard_threshold"]:
            logger.debug(
                "Cross-sectional flatness guard triggered for %s: %.6f < %.6f",
                fp,
                mean_xs_std,
                _EVAL_CONFIG["xs_flatness_guard_threshold"],
            )
            logger.debug(
                "Flatness stats %s | mean_xs_std %.6f (< %.6f)",
                fp,
                mean_xs_std,
                _EVAL_CONFIG["xs_flatness_guard_threshold"],
            )
            score = -float('inf')

    if score > -float('inf'): # Temporal check only if not already penalized to -inf
        if full_raw_predictions_matrix.ndim == 2 and full_raw_predictions_matrix.shape[0] > 1:
            time_std_per_stock = full_raw_predictions_matrix.std(axis=0, ddof=0)
            mean_t_std = np.mean(time_std_per_stock)
            if mean_t_std < _EVAL_CONFIG["temporal_flatness_guard_threshold"]:
                logger.debug(
                    "Temporal flatness guard triggered for %s: %.6f < %.6f",
                    fp,
                    mean_t_std,
                    _EVAL_CONFIG["temporal_flatness_guard_threshold"],
                )
                logger.debug(
                    "Flatness stats %s | mean_t_std %.6f (< %.6f)",
                    fp,
                    mean_t_std,
                    _EVAL_CONFIG["temporal_flatness_guard_threshold"],
                )
                score = -float('inf')
        elif full_raw_predictions_matrix.ndim == 1 and full_raw_predictions_matrix.shape[0] > 1: # Single series case
            mean_t_std = full_raw_predictions_matrix.std(ddof=0)
            if mean_t_std < _EVAL_CONFIG["temporal_flatness_guard_threshold"]:
                logger.debug(
                    "Temporal flatness guard triggered for %s: %.6f < %.6f",
                    fp,
                    mean_t_std,
                    _EVAL_CONFIG["temporal_flatness_guard_threshold"],
                )
                logger.debug(
                    "Flatness stats %s | mean_t_std %.6f (< %.6f)",
                    fp,
                    mean_t_std,
                    _EVAL_CONFIG["temporal_flatness_guard_threshold"],
                )
                score = -float('inf')

    # Flatness guards on processed predictions
    if score > -float('inf'):
        if full_processed_predictions_matrix.ndim == 2 and full_processed_predictions_matrix.shape[1] > 0:
            proc_xs_stds = full_processed_predictions_matrix.std(axis=1, ddof=0)
            mean_proc_xs_std = np.mean(proc_xs_stds)
            if mean_proc_xs_std < _EVAL_CONFIG["xs_flatness_guard_threshold"]:
                logger.debug(
                    "Processed cross-sectional flatness guard triggered for %s: %.6f < %.6f",
                    fp,
                    mean_proc_xs_std,
                    _EVAL_CONFIG["xs_flatness_guard_threshold"],
                )
                logger.debug(
                    "Flatness stats %s | mean_proc_xs_std %.6f (< %.6f)",
                    fp,
                    mean_proc_xs_std,
                    _EVAL_CONFIG["xs_flatness_guard_threshold"],
                )
                score = -float('inf')

    if score > -float('inf'):
        if full_processed_predictions_matrix.ndim == 2 and full_processed_predictions_matrix.shape[0] > 1:
            proc_t_stds = full_processed_predictions_matrix.std(axis=0, ddof=0)
            mean_proc_t_std = np.mean(proc_t_stds)
            if mean_proc_t_std < _EVAL_CONFIG["temporal_flatness_guard_threshold"]:
                logger.debug(
                    "Processed temporal flatness guard triggered for %s: %.6f < %.6f",
                    fp,
                    mean_proc_t_std,
                    _EVAL_CONFIG["temporal_flatness_guard_threshold"],
                )
                logger.debug(
                    "Flatness stats %s | mean_proc_t_std %.6f (< %.6f)",
                    fp,
                    mean_proc_t_std,
                    _EVAL_CONFIG["temporal_flatness_guard_threshold"],
                )
                score = -float('inf')
        elif full_processed_predictions_matrix.ndim == 1 and full_processed_predictions_matrix.shape[0] > 1:
            mean_proc_t_std = full_processed_predictions_matrix.std(ddof=0)
            if mean_proc_t_std < _EVAL_CONFIG["temporal_flatness_guard_threshold"]:
                logger.debug(
                    "Processed temporal flatness guard triggered for %s: %.6f < %.6f",
                    fp,
                    mean_proc_t_std,
                    _EVAL_CONFIG["temporal_flatness_guard_threshold"],
                )
                logger.debug(
                    "Flatness stats %s | mean_proc_t_std %.6f (< %.6f)",
                    fp,
                    mean_proc_t_std,
                    _EVAL_CONFIG["temporal_flatness_guard_threshold"],
                )
                score = -float('inf')

    # HOF correlation penalty (uses processed predictions)
    if score > -float('inf') and processed_for_hof.size > 0:
        correlation_penalty = hof_module.get_correlation_penalty_with_hof(processed_for_hof.flatten())
        if correlation_penalty > 0:
            logger.debug(
                "Correlation penalty for %s: %.6f",
                fp,
                correlation_penalty,
            )
        score -= correlation_penalty
    result = EvalResult(
        score,
        mean_daily_ic,
        sharpe_proxy,
        parsimony_penalty,
        correlation_penalty,
        processed_for_hof if return_preds else None,
        ic_std,
        turnover_proxy,
        None,
    )
    # Compute fixed-weight fitness if candidate wasn't invalidated to -inf
    try:
        if result.fitness > -float("inf") and processed_for_hof.size > 0:
            fixed_sharpe_w = float(_EVAL_CONFIG.get("fixed_sharpe_proxy_weight", 0.0))
            fixed_ic_std_w = float(_EVAL_CONFIG.get("fixed_ic_std_penalty_weight", 0.0))
            fixed_turnover_w = float(_EVAL_CONFIG.get("fixed_turnover_penalty_weight", 0.0))
            fixed_corr_w = float(_EVAL_CONFIG.get("fixed_corr_penalty_weight", 0.0))
            # Use HOF helper at provided weight (cutoff stays as configured in HOF)
            corr_pen_fixed = hof_module.get_correlation_penalty_with_weight(processed_for_hof.flatten(), weight=fixed_corr_w)
            score_fixed = (
                mean_daily_ic
                + fixed_sharpe_w * sharpe_proxy
                - fixed_ic_std_w * ic_std
                - fixed_turnover_w * turnover_proxy
                - parsimony_penalty
                - corr_pen_fixed
            )
            result.fitness_static = float(score_fixed)
    except Exception:
        # Keep fitness_static as None on any failure
        pass
    if result.fitness_static is not None:
        logger.debug(
            "Eval summary %s | fitness %.6f (fixed %.6f) mean_ic %.6f ic_std %.6f turnover %.6f sharpe %.6f parsimony %.6f correlation %.6f ops %d",
            fp,
            result.fitness,
            result.fitness_static,
            result.mean_ic,
            result.ic_std,
            result.turnover_proxy,
            result.sharpe_proxy,
            result.parsimony_penalty,
            result.correlation_penalty,
            prog.size,
        )
    else:
        logger.debug(
            "Eval summary %s | fitness %.6f mean_ic %.6f ic_std %.6f turnover %.6f sharpe %.6f parsimony %.6f correlation %.6f ops %d",
            fp,
            result.fitness,
            result.mean_ic,
            result.ic_std,
            result.turnover_proxy,
            result.sharpe_proxy,
            result.parsimony_penalty,
            result.correlation_penalty,
            prog.size,
        )
    _cache_set(fp, result)
    return result
