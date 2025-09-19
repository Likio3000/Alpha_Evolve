from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Set, Iterable
from collections import OrderedDict
from dataclasses import dataclass, field
import logging

if TYPE_CHECKING:
    from alpha_framework.alpha_framework_program import AlphaProgram # Changed from alpha_program_core
    from alpha_framework.alpha_framework_op import Op # For _uses_feature_vector
    from evolution_components import data_handling # To access data
    from evolution_components import hall_of_fame_manager as hof_manager # To get HOF penalty

from alpha_framework.alpha_framework_types import (
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    FINAL_PREDICTION_VECTOR_NAME,
    SCALAR_FEATURE_NAMES,
)
from backtesting_components.performance_metrics import compute_max_drawdown


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
    factor_penalty: float = 0.0
    # Optional: fixed-weight fitness for comparability across gens (no ramping)
    fitness_static: Optional[float] = None
    horizon_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    max_drawdown: float = 0.0
    factor_exposure_sum: float = 0.0
    robustness_penalty: float = 0.0
    stress_metrics: Dict[str, float] = field(default_factory=dict)


_eval_cache: "OrderedDict[str, EvalResult]" = OrderedDict()
_EVAL_CACHE_MAX_SIZE = 128


def _cache_set(fp: str, value: EvalResult) -> None:
    if fp in _eval_cache:
        _eval_cache.move_to_end(fp)
    elif len(_eval_cache) >= _EVAL_CACHE_MAX_SIZE:
        _eval_cache.popitem(last=False)
    _eval_cache[fp] = value


def _resolve_factor_vector(
    name: str,
    features_at_t: Dict[str, Any],
    aligned_dfs,
    stock_symbols: List[str],
    timestamp,
) -> Optional[np.ndarray]:
    vec_any = features_at_t.get(name)
    if vec_any is not None:
        arr = np.asarray(vec_any, dtype=float)
        if arr.ndim == 1 and arr.size == len(stock_symbols):
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # fallback to dataframe column lookup
    col = name
    if name.endswith("_t"):
        col = name[:-2]
    try:
        arr = np.array(
            [aligned_dfs[sym].loc[timestamp, col] for sym in stock_symbols],
            dtype=float,
        )
        if arr.size == len(stock_symbols):
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None
    return None

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
    "ic_tstat_weight": 0.0,
    # Fixed weights for comparability (no ramping)
    "fixed_sharpe_proxy_weight": 0.0,
    "fixed_ic_std_penalty_weight": 0.0,
    "fixed_turnover_penalty_weight": 0.0,
    "fixed_corr_penalty_weight": 0.0,
    "fixed_ic_tstat_weight": 0.0,
    "factor_penalty_weight": 0.0,
    "factor_penalty_factors": tuple(),
    "stress_penalty_weight": 0.0,
    "stress_fee_bps": 5.0,
    "stress_slippage_bps": 2.0,
    "stress_shock_scale": 1.5,
    "evaluation_horizons": (1,),
    "hof_corr_mode": "flat",   # 'flat' (default) or 'per_bar'
    "temporal_decay_half_life": 0.0,
    "use_train_val_splits": False,
    "train_points": 0,
    "val_points": 0,
    # CPCV-style cross-validation over time (K contiguous folds with embargo)
    "cv_k_folds": 0,
    "cv_embargo": 0,
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
    ic_tstat_weight: float = 0.0,
    factor_penalty_weight: float = 0.0,
    factor_penalty_factors: str | Iterable[int] | None = None,
    stress_penalty_weight: float = 0.0,
    stress_fee_bps: float = 5.0,
    stress_slippage_bps: float = 2.0,
    stress_shock_scale: float = 1.5,
    evaluation_horizons: Iterable[int] | None = None,
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
    fixed_ic_tstat_weight: Optional[float] = None,
    hof_corr_mode: str | None = None,
    temporal_decay_half_life: float | None = None,
    cv_k_folds: int | None = None,
    cv_embargo: int | None = None,
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
    _EVAL_CONFIG["ic_tstat_weight"] = ic_tstat_weight
    _EVAL_CONFIG["factor_penalty_weight"] = float(factor_penalty_weight)
    factors_raw = ""
    factors_tuple: tuple[str, ...]
    if isinstance(factor_penalty_factors, str):
        factors_raw = factor_penalty_factors.strip()
        if factors_raw:
            factors_tuple = tuple(sorted({f.strip() for f in factors_raw.split(',') if f.strip()}))
        else:
            factors_tuple = tuple()
    elif factor_penalty_factors is not None:
        factors_tuple = tuple(sorted({str(f).strip() for f in factor_penalty_factors if str(f).strip()}))
    else:
        factors_tuple = tuple()
    _EVAL_CONFIG["factor_penalty_factors"] = factors_tuple
    _EVAL_CONFIG["stress_penalty_weight"] = float(stress_penalty_weight)
    _EVAL_CONFIG["stress_fee_bps"] = float(stress_fee_bps)
    _EVAL_CONFIG["stress_slippage_bps"] = float(stress_slippage_bps)
    _EVAL_CONFIG["stress_shock_scale"] = float(stress_shock_scale)

    horizons_set: Set[int] = set()
    if evaluation_horizons is not None:
        for val in evaluation_horizons:
            try:
                h = int(val)
            except Exception:
                continue
            if h > 0:
                horizons_set.add(h)
    if not horizons_set:
        horizons_set.add(1)
    _EVAL_CONFIG["evaluation_horizons"] = tuple(sorted(horizons_set))
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
    _EVAL_CONFIG["fixed_ic_tstat_weight"] = (
        float(ic_tstat_weight) if fixed_ic_tstat_weight is None else float(fixed_ic_tstat_weight)
    )
    if hof_corr_mode is not None:
        _EVAL_CONFIG["hof_corr_mode"] = str(hof_corr_mode)
    if temporal_decay_half_life is not None:
        _EVAL_CONFIG["temporal_decay_half_life"] = float(temporal_decay_half_life)
    if cv_k_folds is not None:
        _EVAL_CONFIG["cv_k_folds"] = int(cv_k_folds)
    if cv_embargo is not None:
        _EVAL_CONFIG["cv_embargo"] = int(cv_embargo)
    # Clamp jitter into [0, 1] and store
    try:
        pj = float(parsimony_jitter_pct)
    except Exception:
        pj = 0.0
    _EVAL_CONFIG["parsimony_jitter_pct"] = max(0.0, min(1.0, pj))
    logging.getLogger(__name__).debug(
        "Evaluation configured: scale=%s parsimony=%s sharpe_w=%s ic_std_w=%s turnover_w=%s factor_w=%s factors=%s horizons=%s splits=%s train=%s val=%s sector_neutralize=%s winsor_p=%.3f jitter=%.3f",
        scale_method,
        parsimony_penalty,
        sharpe_proxy_weight,
        ic_std_penalty_weight,
        turnover_penalty_weight,
        _EVAL_CONFIG["factor_penalty_weight"],
        _EVAL_CONFIG["factor_penalty_factors"],
        _EVAL_CONFIG["evaluation_horizons"],
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
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    # Center and compute via dot product for speed/stability
    a = a.astype(float, copy=False)
    b = b.astype(float, copy=False)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    if not np.isfinite(denom) or denom < 1e-9:
        return 0.0
    corr = float(np.dot(a, b) / denom)
    # Guard tiny numerical drift
    if not np.isfinite(corr):
        return 0.0
    return max(-1.0, min(1.0, corr))


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

    Vectorized implementation using indexed accumulation for speed.
    """
    if x.size == 0:
        return x
    g_raw = groups.astype(int, copy=False)
    # Compact group ids to 0..G-1
    uniq, inv = np.unique(g_raw, return_inverse=True)
    g = inv
    G = uniq.size
    sums = np.zeros(G, dtype=float)
    counts = np.zeros(G, dtype=float)
    np.add.at(sums, g, x)
    np.add.at(counts, g, 1.0)
    means = np.zeros(G, dtype=float)
    np.divide(sums, counts, out=means, where=counts > 0)
    out = x - means[g]
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
        result = EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0)
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
        # Context-first migration: prefer explicit EvalContext.
        # Fall back to module globals for compatibility (warn once at INFO).
        try:
            _warned = globals().get("_CTXLESS_WARNED", False)
            if not _warned:
                logging.getLogger(__name__).info(
                    "evaluate_program called without EvalContext; falling back to globals."
                )
                globals()["_CTXLESS_WARNED"] = True
        except Exception:
            pass
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
    daily_pnl_values: List[float] = []

    horizons_cfg = _EVAL_CONFIG.get("evaluation_horizons", (eval_lag,))
    horizons: List[int] = []
    try:
        for h in horizons_cfg:
            h_int = int(h)
            if h_int > 0:
                horizons.append(h_int)
    except Exception:
        horizons = []
    if not horizons:
        horizons = [eval_lag if eval_lag > 0 else 1]
    if eval_lag > 0 and eval_lag not in horizons:
        horizons.append(eval_lag)
    horizons = sorted(set(horizons))
    primary_horizon = eval_lag if eval_lag in horizons else horizons[0]
    max_horizon = max(horizons)

    close_matrix: Optional[np.ndarray] = None
    has_close_prices = False
    if ctx is not None and getattr(ctx, "col_matrix_map", None):
        for key in ("close", "closes", "close_adj"):
            mat = ctx.col_matrix_map.get(key)  # type: ignore[union-attr]
            if mat is not None and mat.shape[0] == len(common_time_index):
                close_matrix = np.array(mat, dtype=float, copy=True)
                break
    if close_matrix is None:
        close_matrix = np.zeros((len(common_time_index), n_stocks), dtype=float)
        for j, sym in enumerate(stock_symbols):
            try:
                series = aligned_dfs[sym]["close"].reindex(common_time_index)
                close_matrix[:, j] = np.nan_to_num(series.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                close_matrix[:, j] = 0.0
    if close_matrix.size:
        has_close_prices = bool(np.any(np.abs(close_matrix) > 1e-9))

    ret_fwd_matrix: Optional[np.ndarray] = None
    if ctx is not None and getattr(ctx, "col_matrix_map", None):
        mat = ctx.col_matrix_map.get("ret_fwd")  # type: ignore[union-attr]
        if mat is None:
            mat = ctx.col_matrix_map.get("ret_fwd_t")  # type: ignore[union-attr]
        if mat is not None and mat.shape[0] == len(common_time_index):
            ret_fwd_matrix = np.array(mat, dtype=float, copy=True)
    if ret_fwd_matrix is None and n_stocks > 0:
        ret_columns: list[np.ndarray] = []
        any_ret_data = False
        for sym in stock_symbols:
            arr = np.zeros(len(common_time_index), dtype=float)
            try:
                series = aligned_dfs[sym]["ret_fwd"].reindex(common_time_index)
                arr = series.to_numpy(dtype=float)
                any_ret_data = True
            except Exception:
                pass
            ret_columns.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        if any_ret_data and ret_columns:
            ret_fwd_matrix = np.column_stack(ret_columns)

    returns_by_horizon: Dict[int, np.ndarray] = {}
    T_total = close_matrix.shape[0]
    for h in horizons:
        if h >= T_total:
            continue
        numer = close_matrix[h:, :]
        denom = close_matrix[: T_total - h, :]
        ret = np.zeros_like(denom)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(numer, denom, out=ret, where=np.abs(denom) > 1e-12)
        ret = np.nan_to_num(ret - 1.0, nan=0.0, posinf=0.0, neginf=0.0)
        returns_by_horizon[h] = ret

    num_evaluation_steps = len(common_time_index) - max_horizon
    if num_evaluation_steps <= 0:  # Not enough data for even one evaluation
        _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0))
        return _eval_cache[fp]

    for h in list(returns_by_horizon.keys()):
        mat = returns_by_horizon[h]
        if mat.shape[0] > num_evaluation_steps:
            returns_by_horizon[h] = mat[:num_evaluation_steps]

    if ret_fwd_matrix is not None and num_evaluation_steps > 0:
        ret_fwd_trim = ret_fwd_matrix[:num_evaluation_steps]
        ret_fwd_trim = np.nan_to_num(ret_fwd_trim, nan=0.0, posinf=0.0, neginf=0.0)
        target_h = eval_lag if eval_lag in horizons else horizons[0]
        if not has_close_prices or target_h not in returns_by_horizon or returns_by_horizon[target_h].size == 0:
            returns_by_horizon[target_h] = ret_fwd_trim

    ic_values_by_h: Dict[int, List[float]] = {h: [] for h in horizons}
    pnl_values_by_h: Dict[int, List[float]] = {h: [] for h in horizons}

    factor_penalty_weight = float(_EVAL_CONFIG.get("factor_penalty_weight", 0.0) or 0.0)
    factor_names_cfg = _EVAL_CONFIG.get("factor_penalty_factors", tuple())
    factor_names: tuple[str, ...]
    if isinstance(factor_names_cfg, (list, tuple, set)):
        factor_names = tuple(str(n) for n in factor_names_cfg if str(n))
    else:
        factor_names = tuple()
    factor_corr_tracker: Dict[str, List[float]] = {name: [] for name in factor_names}

    # The common_time_index from data_handling is already sliced appropriately for eval_lag
    # Loop until common_time_index.size - eval_lag -1 (for python 0-indexing)
    # Or more simply, loop `len(common_time_index) - eval_lag` times.
    # If common_time_index has N points, we can make N - eval_lag predictions.
    # Prediction at t_idx uses features at t_idx, compares with returns at t_idx + eval_lag.
    
    for t_idx in range(num_evaluation_steps):
        timestamp = common_time_index[t_idx]

        # Prefer fast-path from precomputed matrices if provided via context
        if ctx is not None and getattr(ctx, "col_matrix_map", None):
            features_at_t: Dict[str, Any] = {}
            for feat_name_template in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
                if feat_name_template == "sector_id_vector":
                    features_at_t[feat_name_template] = sector_groups_vec
                    continue
                col = feat_name_template.replace("_t", "")
                mat = ctx.col_matrix_map.get(col)  # type: ignore[union-attr]
                if mat is not None and 0 <= t_idx < mat.shape[0]:
                    features_at_t[feat_name_template] = np.nan_to_num(mat[t_idx, :], nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    # Fallback to dataframe lookup for missing columns
                    try:
                        vec = np.array([
                            aligned_dfs[sym].loc[timestamp, col] for sym in stock_symbols
                        ], dtype=float)
                        features_at_t[feat_name_template] = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        features_at_t[feat_name_template] = np.zeros(len(stock_symbols))
            # Scalars
            for sc in SCALAR_FEATURE_NAMES:
                if sc == "const_1":
                    features_at_t[sc] = 1.0
                elif sc == "const_neg_1":
                    features_at_t[sc] = -1.0
        else:
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
                _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0))
                return _eval_cache[fp]
            
            all_raw_predictions_timeseries.append(raw_predictions_t.copy())
            
            processed_predictions_t = _scale_signal_for_ic(
                raw_predictions_t, _EVAL_CONFIG["ic_scale_method"]
            )
            if _EVAL_CONFIG.get("sector_neutralize", False):
                processed_predictions_t = _demean_by_groups(processed_predictions_t, sector_groups_vec)
            all_processed_predictions_timeseries.append(processed_predictions_t)

            if factor_corr_tracker:
                pred_centered = processed_predictions_t - float(np.mean(processed_predictions_t))
                pred_std = float(np.std(pred_centered, ddof=0))
                if pred_std > 1e-9:
                    pred_z = pred_centered / pred_std
                    for fname in factor_corr_tracker.keys():
                        vec = _resolve_factor_vector(fname, features_at_t, aligned_dfs, stock_symbols, timestamp)
                        if vec is None or vec.size != pred_z.size:
                            continue
                        vec_centered = vec - float(np.mean(vec))
                        vec_std = float(np.std(vec_centered, ddof=0))
                        if vec_std <= 1e-9:
                            continue
                        vec_z = vec_centered / vec_std
                        corr = float(np.dot(pred_z, vec_z) / pred_z.size)
                        if np.isfinite(corr):
                            factor_corr_tracker[fname].append(corr)

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
                    result = EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0)
                    _cache_set(fp, result)
                    return result

            # IC calculation
            for horizon in horizons:
                ret_matrix = returns_by_horizon.get(horizon)
                if ret_matrix is None or t_idx >= ret_matrix.shape[0]:
                    continue
                actual_returns_t = ret_matrix[t_idx]
                if np.all(np.isnan(actual_returns_t)):
                    ic_value = 0.0
                    pnl_val = 0.0
                else:
                    r_pred = _average_rank_ties(processed_predictions_t)
                    r_rets = _average_rank_ties(actual_returns_t)
                    ic_t = _safe_corr_eval(r_pred, r_rets)
                    ic_value = 0.0 if np.isnan(ic_t) else ic_t
                    pnl_val = float(np.mean(processed_predictions_t * actual_returns_t))
                ic_values_by_h[horizon].append(ic_value)
                pnl_values_by_h[horizon].append(pnl_val)
                if horizon == primary_horizon:
                    daily_ic_values.append(ic_value)
                    daily_pnl_values.append(pnl_val)

        except Exception: # Broad exception during program's .eval() or IC calculation
            _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0))
            return _eval_cache[fp]

    if not daily_ic_values or not all_raw_predictions_timeseries or not all_processed_predictions_timeseries:
        _push_event({"fp": fp, "event": "exception"})
        _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0))
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
    split_weighting = str(_EVAL_CONFIG.get("split_weighting", "equal")) if _EVAL_CONFIG.get("split_weighting") is not None else "equal"

    # Optional temporal decay weighting for mean IC (recent bars matter more)
    def _decay_weights(T: int, half_life: float) -> np.ndarray:
        if T <= 0 or half_life <= 0.0:
            return np.ones(T, dtype=float)
        ages = np.arange(T-1, -1, -1, dtype=float)  # 0 for newest bar
        return 0.5 ** (ages / float(half_life))

    hl = float(_EVAL_CONFIG.get("temporal_decay_half_life", 0.0) or 0.0)
    if not use_splits and hl > 0.0 and (int(_EVAL_CONFIG.get("cv_k_folds", 0) or 0) <= 1):
        T = len(daily_ic_values)
        w = _decay_weights(T, hl)
        ws = np.sum(w) if T > 0 else 1.0
        mean_daily_ic = float(np.sum(w * np.array(daily_ic_values)) / (ws if ws > 0 else 1.0))
    else:
        mean_daily_ic = float(np.mean(daily_ic_values))
    pnl_primary = np.array(pnl_values_by_h.get(primary_horizon, []), dtype=float)
    if pnl_primary.size > 0:
        mean_pnl = float(np.mean(pnl_primary))
        std_pnl = float(np.std(pnl_primary, ddof=0))
        sharpe_proxy = mean_pnl / std_pnl if std_pnl > 1e-9 else 0.0
    else:
        mean_pnl = 0.0
        std_pnl = 0.0
        sharpe_proxy = 0.0
    ic_std = float(np.std(daily_ic_values, ddof=0)) if daily_ic_values else 0.0
    turnover_proxy = 0.0

    cv_k = int(_EVAL_CONFIG.get("cv_k_folds", 0) or 0)
    cv_emb = int(_EVAL_CONFIG.get("cv_embargo", 0) or 0)

    if cv_k > 1 and total_steps > cv_k:
        # Combinatorial (contiguous) purged CV: split into K folds, compute validation metrics
        fold_len = max(1, total_steps // cv_k)
        ic_vals = []
        ic_stds = []
        turns = []
        shs = []
        # Collect processed predictions on validation segments for HOF/penalties
        val_pred_segs: list[np.ndarray] = []
        for f in range(cv_k):
            start = f * fold_len
            end = total_steps if f == cv_k - 1 else (f + 1) * fold_len
            # Validation slice
            va = slice(start, end)
            # Embargo region around validation slice (excluded from training, irrelevant here since we only need val metrics)
            # Compute IC mean (optionally with decay within fold)
            arr_va = np.array(daily_ic_values[va], dtype=float)
            if arr_va.size == 0:
                continue
            if hl > 0.0:
                w_va = _decay_weights(arr_va.size, hl)
                ic_va = float(np.sum(w_va * arr_va) / (np.sum(w_va) if np.sum(w_va) > 0 else 1.0))
            else:
                ic_va = float(np.mean(arr_va))
            ic_vals.append(ic_va)
            ic_stds.append(float(np.std(arr_va, ddof=0)))
            # Turnover on validation positions
            pos_va = _neutralize(full_processed_predictions_matrix[va])
            if pos_va.shape[0] > 1:
                per_step_va = np.sum(np.abs(np.diff(pos_va, axis=0)), axis=1)/2.0
                if hl > 0.0:
                    wt_va = _decay_weights(per_step_va.size + 1, hl)[1:]
                    turn_va = float(np.sum(wt_va * per_step_va) / (np.sum(wt_va) if np.sum(wt_va) > 0 else 1.0))
                else:
                    turn_va = float(np.mean(per_step_va))
            else:
                turn_va = 0.0
            turns.append(turn_va)
            # Sharpe proxy on validation pnl
            pnl_va = np.array(daily_pnl_values[va], dtype=float)
            if pnl_va.size > 1:
                m = float(np.mean(pnl_va))
                s = float(np.std(pnl_va, ddof=0))
                sharpe_va = float(m / s) if s > 1e-12 else 0.0
            else:
                sharpe_va = 0.0
            shs.append(sharpe_va)
            # Accumulate processed preds for HOF correlation penalty
            try:
                seg = full_processed_predictions_matrix[va]
                if seg.size > 0:
                    val_pred_segs.append(seg)
            except Exception:
                pass
        # Aggregate across folds (simple mean)
        mean_daily_ic = float(np.mean(ic_vals)) if ic_vals else mean_daily_ic
        ic_std = float(np.mean(ic_stds)) if ic_stds else ic_std
        turnover_proxy = float(np.mean(turns)) if turns else turnover_proxy
        sharpe_proxy = float(np.mean(shs)) if shs else sharpe_proxy
        processed_for_hof = np.vstack(val_pred_segs) if val_pred_segs else full_processed_predictions_matrix
    elif use_splits and t_points > 0 and v_points > 0 and (t_points + v_points) <= total_steps:
        tr = slice(0, t_points)
        va = slice(t_points, t_points + v_points)
        if hl > 0.0:
            arr_tr = np.array(daily_ic_values[tr], dtype=float)
            arr_va = np.array(daily_ic_values[va], dtype=float)
            w_tr = _decay_weights(arr_tr.size, hl)
            w_va = _decay_weights(arr_va.size, hl)
            ic_tr = float(np.sum(w_tr * arr_tr) / (np.sum(w_tr) if np.sum(w_tr) > 0 else 1.0)) if t_points > 0 else 0.0
            ic_va = float(np.sum(w_va * arr_va) / (np.sum(w_va) if np.sum(w_va) > 0 else 1.0)) if v_points > 0 else 0.0
        else:
            ic_tr = float(np.mean(daily_ic_values[tr])) if t_points > 0 else 0.0
            ic_va = float(np.mean(daily_ic_values[va])) if v_points > 0 else 0.0
        ic_std_tr = float(np.std(daily_ic_values[tr], ddof=0)) if t_points > 1 else 0.0
        ic_std_va = float(np.std(daily_ic_values[va], ddof=0)) if v_points > 1 else 0.0

        pos_tr = _neutralize(full_processed_predictions_matrix[tr])
        pos_va = _neutralize(full_processed_predictions_matrix[va])
        if pos_tr.shape[0] > 1:
            per_step_tr = np.sum(np.abs(np.diff(pos_tr, axis=0)), axis=1)/2.0
            if hl > 0.0:
                wt_tr = _decay_weights(per_step_tr.size + 1, hl)[1:]
                turn_tr = float(np.sum(wt_tr * per_step_tr) / (np.sum(wt_tr) if np.sum(wt_tr) > 0 else 1.0))
            else:
                turn_tr = float(np.mean(per_step_tr))
        else:
            turn_tr = 0.0
        if pos_va.shape[0] > 1:
            per_step_va = np.sum(np.abs(np.diff(pos_va, axis=0)), axis=1)/2.0
            if hl > 0.0:
                wt_va = _decay_weights(per_step_va.size + 1, hl)[1:]
                turn_va = float(np.sum(wt_va * per_step_va) / (np.sum(wt_va) if np.sum(wt_va) > 0 else 1.0))
            else:
                turn_va = float(np.mean(per_step_va))
        else:
            turn_va = 0.0

        if split_weighting == "by_points":
            w_tr = float(t_points)
            w_va = float(v_points)
            w_sum = max(1.0, w_tr + w_va)
            mean_daily_ic = (w_tr * ic_tr + w_va * ic_va) / w_sum
            ic_std = (w_tr * ic_std_tr + w_va * ic_std_va) / w_sum
            turnover_proxy = (w_tr * turn_tr + w_va * turn_va) / w_sum
        else:
            mean_daily_ic = 0.5 * (ic_tr + ic_va)
            ic_std = 0.5 * (ic_std_tr + ic_std_va)
            turnover_proxy = 0.5 * (turn_tr + turn_va)
        processed_for_hof = full_processed_predictions_matrix[va]
    else:
        pos_full = _neutralize(full_processed_predictions_matrix)
        if pos_full.shape[0] > 1:
            per_step_turn = np.sum(np.abs(np.diff(pos_full, axis=0)), axis=1)/2.0
            if hl > 0.0 and not use_splits:
                wt = _decay_weights(per_step_turn.size + 1, hl)[1:]
                wts = float(np.sum(wt))
                turnover_proxy = float(np.sum(wt * per_step_turn) / (wts if wts > 0 else 1.0))
            else:
                turnover_proxy = float(np.mean(per_step_turn))
        else:
            turnover_proxy = 0.0
        processed_for_hof = full_processed_predictions_matrix

    horizon_metrics: Dict[int, Dict[str, float]] = {}
    combined_ic_values = None
    if len(horizons) > 1:
        ic_means_components: List[float] = []
        ic_std_components: List[float] = []
        sharpe_components: List[float] = []
        combined_segments: List[np.ndarray] = []
        mean_pnl_components: List[float] = []
        for h in horizons:
            ic_vals = np.array(ic_values_by_h.get(h, []), dtype=float)
            pnl_vals = np.array(pnl_values_by_h.get(h, []), dtype=float)
            mean_ic_h = float(np.mean(ic_vals)) if ic_vals.size else 0.0
            ic_std_h = float(np.std(ic_vals, ddof=0)) if ic_vals.size else 0.0
            mean_pnl_h = float(np.mean(pnl_vals)) if pnl_vals.size else 0.0
            if pnl_vals.size > 1:
                sharpe_h = float(mean_pnl_h / np.std(pnl_vals, ddof=0)) if np.std(pnl_vals, ddof=0) > 1e-9 else 0.0
            else:
                sharpe_h = 0.0
            horizon_metrics[h] = {
                "mean_ic": mean_ic_h,
                "ic_std": ic_std_h,
                "mean_pnl": mean_pnl_h,
                "sharpe": sharpe_h,
            }
            if ic_vals.size:
                ic_means_components.append(mean_ic_h)
                ic_std_components.append(ic_std_h)
                combined_segments.append(ic_vals)
            if pnl_vals.size:
                mean_pnl_components.append(mean_pnl_h)
            sharpe_components.append(sharpe_h)
        if ic_means_components:
            mean_daily_ic = float(np.mean(ic_means_components))
        if ic_std_components:
            ic_std = float(np.mean(ic_std_components))
        if sharpe_components:
            sharpe_proxy = float(np.mean(sharpe_components))
        if mean_pnl_components:
            mean_pnl = float(np.mean(mean_pnl_components))
        if combined_segments:
            combined_ic_values = np.concatenate(combined_segments)
    else:
        horizon_metrics[primary_horizon] = {
            "mean_ic": mean_daily_ic,
            "ic_std": ic_std,
            "mean_pnl": mean_pnl,
            "sharpe": sharpe_proxy,
        }
        combined_ic_values = np.array(daily_ic_values, dtype=float)

    drawdown_proxy = 0.0
    for horizon in horizons:
        pnl_series = np.array(pnl_values_by_h.get(horizon, []), dtype=float)
        dd_value = 0.0
        if pnl_series.size:
            equity_curve = np.cumsum(pnl_series)
            dd_value = float(compute_max_drawdown(equity_curve))
        bucket = horizon_metrics.setdefault(horizon, {})
        bucket["max_drawdown"] = dd_value
        if horizon == primary_horizon:
            drawdown_proxy = dd_value

    stress_metrics: Dict[str, float] = {}
    robustness_penalty = 0.0
    if pnl_primary.size > 0:
        fee_bps = float(_EVAL_CONFIG.get("stress_fee_bps", 0.0) or 0.0)
        slip_bps = float(_EVAL_CONFIG.get("stress_slippage_bps", 0.0) or 0.0)
        fee_total = (fee_bps + slip_bps) / 10000.0
        turnover_est = max(0.0, float(turnover_proxy))
        stress_cost = fee_total * turnover_est
        shock_scale = float(max(1.0, float(_EVAL_CONFIG.get("stress_shock_scale", 1.0) or 1.0)))
        stressed = pnl_primary.astype(float, copy=True)
        if stressed.size > 0:
            neg_mask = stressed < 0.0
            pos_mask = stressed > 0.0
            stressed[neg_mask] *= shock_scale
            if shock_scale > 1.0:
                stressed[pos_mask] /= shock_scale
        stress_equity = np.cumsum(stressed)
        stress_dd = float(compute_max_drawdown(stress_equity))
        stress_mean = float(np.mean(stressed)) - stress_cost
        stress_metrics = {
            "cost": float(stress_cost),
            "drawdown": stress_dd,
            "mean_pnl": stress_mean,
            "shock_scale": shock_scale,
        }
        if stress_cost > 0.0:
            robustness_penalty += stress_cost
        if stress_dd > 0.0:
            robustness_penalty += stress_dd
        if stress_mean < 0.0:
            robustness_penalty += abs(stress_mean)

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
    # IC t-statistic component (optional)
    ic_tstat = 0.0
    try:
        arr_ic: Optional[np.ndarray] = None
        weighted = False
        if combined_ic_values is not None and len(combined_ic_values) > 0:
            arr_ic = np.array(combined_ic_values, dtype=float)
        elif daily_ic_values:
            arr_ic = np.array(daily_ic_values, dtype=float)
            weighted = not use_splits and hl > 0.0 and (int(_EVAL_CONFIG.get("cv_k_folds", 0) or 0) <= 1)
        if arr_ic is not None and arr_ic.size:
            if weighted and arr_ic.size == len(daily_ic_values):
                w = _decay_weights(arr_ic.size, hl)
                mu_w = float(np.sum(w * arr_ic) / (np.sum(w) if np.sum(w) > 0 else 1.0))
                wsum = float(np.sum(w))
                w2sum = float(np.sum(w * w))
                neff = wsum * wsum / (w2sum if w2sum > 0 else 1.0)
                s = float(np.std(arr_ic, ddof=0))
                ic_tstat = float(mu_w / (s / np.sqrt(max(1.0, neff)))) if s > 1e-12 else 0.0
            else:
                mu = float(np.mean(arr_ic))
                s = float(np.std(arr_ic, ddof=0))
                n = float(arr_ic.size)
                ic_tstat = float(mu / (s / np.sqrt(n))) if s > 1e-12 and n > 1 else 0.0
    except Exception:
        ic_tstat = 0.0

    score = (
        mean_daily_ic
        + _EVAL_CONFIG.get("sharpe_proxy_weight", 0.0) * sharpe_proxy
        - _EVAL_CONFIG.get("ic_std_penalty_weight", 0.0) * ic_std
        - _EVAL_CONFIG.get("turnover_penalty_weight", 0.0) * turnover_proxy
        - parsimony_penalty
        + _EVAL_CONFIG.get("ic_tstat_weight", 0.0) * ic_tstat
    )
    correlation_penalty = 0.0

    factor_penalty_components: Dict[str, float] = {}
    if factor_corr_tracker:
        for name, values in factor_corr_tracker.items():
            if not values:
                continue
            mean_abs_corr = float(np.mean(np.abs(values)))
            factor_penalty_components[name] = mean_abs_corr
    factor_exposure_sum = float(sum(factor_penalty_components.values())) if factor_penalty_components else 0.0
    factor_penalty_value = 0.0
    if factor_penalty_weight > 0.0 and factor_penalty_components:
        factor_penalty_value = factor_penalty_weight * factor_exposure_sum
        score -= factor_penalty_value

    stress_penalty_weight = float(_EVAL_CONFIG.get("stress_penalty_weight", 0.0) or 0.0)
    if stress_penalty_weight > 0.0 and robustness_penalty > 0.0:
        score -= stress_penalty_weight * robustness_penalty

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
        mode = str(_EVAL_CONFIG.get("hof_corr_mode", "flat"))
        if mode == "per_bar" and processed_for_hof.ndim == 2 and processed_for_hof.shape[0] > 0:
            correlation_penalty = hof_module.get_correlation_penalty_per_bar(processed_for_hof)
        else:
            correlation_penalty = hof_module.get_correlation_penalty_with_hof(processed_for_hof.flatten())
        if correlation_penalty > 0:
            logger.debug(
                "Correlation penalty for %s: %.6f",
                fp,
                correlation_penalty,
            )
        score -= correlation_penalty
    result = EvalResult(
        fitness=score,
        mean_ic=mean_daily_ic,
        sharpe_proxy=sharpe_proxy,
        parsimony_penalty=parsimony_penalty,
        correlation_penalty=correlation_penalty,
        processed_predictions=processed_for_hof if return_preds else None,
        ic_std=ic_std,
        turnover_proxy=turnover_proxy,
        factor_penalty=factor_penalty_value,
        fitness_static=None,
        horizon_metrics=horizon_metrics,
        factor_exposures=factor_penalty_components,
        max_drawdown=drawdown_proxy,
        factor_exposure_sum=factor_exposure_sum,
        robustness_penalty=robustness_penalty,
        stress_metrics=stress_metrics,
    )
    # Compute fixed-weight fitness if candidate wasn't invalidated to -inf
    try:
        if result.fitness > -float("inf") and processed_for_hof.size > 0:
            fixed_sharpe_w = float(_EVAL_CONFIG.get("fixed_sharpe_proxy_weight", 0.0))
            fixed_ic_std_w = float(_EVAL_CONFIG.get("fixed_ic_std_penalty_weight", 0.0))
            fixed_turnover_w = float(_EVAL_CONFIG.get("fixed_turnover_penalty_weight", 0.0))
            fixed_corr_w = float(_EVAL_CONFIG.get("fixed_corr_penalty_weight", 0.0))
            # Use HOF helper at provided weight (cutoff stays as configured in HOF)
            mode = str(_EVAL_CONFIG.get("hof_corr_mode", "flat"))
            if mode == "per_bar" and processed_for_hof.ndim == 2 and processed_for_hof.shape[0] > 0:
                corr_pen_fixed = hof_module.get_correlation_penalty_with_weight_per_bar(processed_for_hof, weight=fixed_corr_w)
            else:
                corr_pen_fixed = hof_module.get_correlation_penalty_with_weight(processed_for_hof.flatten(), weight=fixed_corr_w)
            score_fixed = (
                mean_daily_ic
                + fixed_sharpe_w * sharpe_proxy
                - fixed_ic_std_w * ic_std
                - fixed_turnover_w * turnover_proxy
                - parsimony_penalty
                - corr_pen_fixed
            )
            # Include IC t-stat in fixed fitness at configured fixed weight
            fixed_ic_t_w = float(_EVAL_CONFIG.get("fixed_ic_tstat_weight", 0.0))
            if fixed_ic_t_w != 0.0:
                score_fixed += fixed_ic_t_w * ic_tstat
            result.fitness_static = float(score_fixed)
    except Exception:
        # Keep fitness_static as None on any failure
        pass
    if result.fitness_static is not None:
        logger.debug(
            "Eval summary %s | fitness %.6f (fixed %.6f) mean_ic %.6f ic_std %.6f turnover %.6f sharpe %.6f parsimony %.6f correlation %.6f factor %.6f drawdown %.6f factor_sum %.6f robust %.6f stress %s ops %d factors %s horizons %s",
            fp,
            result.fitness,
            result.fitness_static,
            result.mean_ic,
            result.ic_std,
            result.turnover_proxy,
            result.sharpe_proxy,
            result.parsimony_penalty,
            result.correlation_penalty,
            result.factor_penalty,
            result.max_drawdown,
            result.factor_exposure_sum,
            result.robustness_penalty,
            result.stress_metrics,
            prog.size,
            factor_penalty_components,
            horizon_metrics,
        )
    else:
        logger.debug(
            "Eval summary %s | fitness %.6f mean_ic %.6f ic_std %.6f turnover %.6f sharpe %.6f parsimony %.6f correlation %.6f factor %.6f drawdown %.6f factor_sum %.6f robust %.6f stress %s ops %d factors %s horizons %s",
            fp,
            result.fitness,
            result.mean_ic,
            result.ic_std,
            result.turnover_proxy,
            result.sharpe_proxy,
            result.parsimony_penalty,
            result.correlation_penalty,
            result.factor_penalty,
            result.max_drawdown,
            result.factor_exposure_sum,
            result.robustness_penalty,
            result.stress_metrics,
            prog.size,
            factor_penalty_components,
            horizon_metrics,
        )
    _cache_set(fp, result)
    return result
