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

from alpha_framework.alpha_framework_types import ( # Ensure these are correct based on where AlphaProgram is defined
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    SCALAR_FEATURE_NAMES,
    FINAL_PREDICTION_VECTOR_NAME
)


# Module-level cache (least-recently used)
@dataclass
class EvalResult:
    fitness: float
    mean_ic: float
    parsimony_penalty: float
    correlation_penalty: float
    processed_predictions: Optional[np.ndarray]


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
    "ic_scale_method": "zscore", # from args.scale
    # EVAL_LAG is handled by data_handling module ensuring data is sliced appropriately
}

def configure_evaluation(
    parsimony_penalty: float, 
    max_ops: int, 
    xs_flatness_guard: float,
    temporal_flatness_guard: float,
    early_abort_bars: int,
    early_abort_xs: float,
    early_abort_t: float,
    scale_method: str
    ):
    global _EVAL_CONFIG
    _EVAL_CONFIG["parsimony_penalty_factor"] = parsimony_penalty
    _EVAL_CONFIG["max_ops_for_parsimony"] = max_ops
    _EVAL_CONFIG["xs_flatness_guard_threshold"] = xs_flatness_guard
    _EVAL_CONFIG["temporal_flatness_guard_threshold"] = temporal_flatness_guard
    _EVAL_CONFIG["early_abort_bars"] = early_abort_bars
    _EVAL_CONFIG["early_abort_xs_threshold"] = early_abort_xs
    _EVAL_CONFIG["early_abort_t_threshold"] = early_abort_t
    _EVAL_CONFIG["ic_scale_method"] = scale_method
    logging.getLogger(__name__).debug(
        "Evaluation logic configured: %s, %s", scale_method, parsimony_penalty
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


def evaluate_program(
    prog: AlphaProgram,
    dh_module: data_handling, # Pass the data_handling module for access
    hof_module: hof_manager,  # Pass the hall_of_fame_manager module
    initial_prog_state_vars_config: Dict[str, Any] # e.g. evolve_alphas.INITIAL_STATE_VARS
) -> EvalResult:
    # Uses _EVAL_CONFIG for various thresholds and penalties

    logger = logging.getLogger(__name__)

    fp = prog.fingerprint
    if fp in _eval_cache:
        _eval_cache.move_to_end(fp)
        cached = _eval_cache[fp]
        logger.debug("Cache hit for %s with fitness %.6f", fp, cached.fitness)
        return cached

    if not _uses_feature_vector_check(prog):
        logger.debug("Program %s does not use any feature vector", fp)
        result = EvalResult(-float('inf'), 0.0, 0.0, 0.0, None)
        _cache_set(fp, result)
        return result

    # Get data from data_handling module
    aligned_dfs = dh_module.get_aligned_dfs()
    common_time_index = dh_module.get_common_time_index()
    stock_symbols = dh_module.get_stock_symbols()
    n_stocks = dh_module.get_n_stocks()
    eval_lag = dh_module.get_eval_lag() # Get eval_lag from data_handling

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

    # The common_time_index from data_handling is already sliced appropriately for eval_lag
    # Loop until common_time_index.size - eval_lag -1 (for python 0-indexing)
    # Or more simply, loop `len(common_time_index) - eval_lag` times.
    # If common_time_index has N points, we can make N - eval_lag predictions.
    # Prediction at t_idx uses features at t_idx, compares with returns at t_idx + eval_lag.
    
    num_evaluation_steps = len(common_time_index) - eval_lag
    if num_evaluation_steps <= 0: # Not enough data for even one evaluation
        _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, None))
        return _eval_cache[fp]

    for t_idx in range(num_evaluation_steps):
        timestamp = common_time_index[t_idx]

        features_at_t: Dict[str, Any] = {}
        for feat_name_template in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
            if feat_name_template == "sector_id_vector":
                features_at_t[feat_name_template] = sector_groups_vec
                continue
            col_name = feat_name_template.replace('_t', '')
            try:
                feat_vec = np.array([aligned_dfs[sym].loc[timestamp, col_name] for sym in stock_symbols], dtype=float)
                features_at_t[feat_name_template] = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)  # Clean at source
            except KeyError:
                features_at_t[feat_name_template] = np.zeros(n_stocks, dtype=float)
            except Exception:  # Broad exception during feature gathering
                features_at_t[feat_name_template] = np.zeros(n_stocks, dtype=float)


        for sc_name in SCALAR_FEATURE_NAMES:
            if sc_name == "const_1":
                features_at_t[sc_name] = 1.0
            elif sc_name == "const_neg_1":
                features_at_t[sc_name] = -1.0
            # Add other scalar features if any

        try:
            # prog.eval uses n_stocks for internal vector shaping/broadcasting.
            raw_predictions_t = prog.eval(features_at_t, program_state, n_stocks)

            if np.any(np.isnan(raw_predictions_t)) or np.any(np.isinf(raw_predictions_t)):
                # print(f"Program {fp} yielded NaN/Inf at t_idx {t_idx}. Aborting eval for this prog.")
                _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, None))
                return _eval_cache[fp]
            
            all_raw_predictions_timeseries.append(raw_predictions_t.copy())
            
            processed_predictions_t = _scale_signal_for_ic(
                raw_predictions_t, _EVAL_CONFIG["ic_scale_method"]
            )
            all_processed_predictions_timeseries.append(processed_predictions_t)

            # Early abort checks (on raw predictions)
            if len(all_raw_predictions_timeseries) == _EVAL_CONFIG["early_abort_bars"]:
                partial_raw_preds_matrix = np.array(all_raw_predictions_timeseries)
                mean_xs_std_partial = 0.0
                if partial_raw_preds_matrix.ndim == 2 and partial_raw_preds_matrix.shape[1] > 0:
                    mean_xs_std_partial = np.mean(partial_raw_preds_matrix.std(axis=1, ddof=0))
                
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
                    early_abort = True
                if mean_t_std_partial < _EVAL_CONFIG["early_abort_t_threshold"]:
                    logger.debug(
                        "Early abort for %s due to temporal flatness %.6f < %.6f",
                        fp,
                        mean_t_std_partial,
                        _EVAL_CONFIG["early_abort_t_threshold"],
                    )
                    early_abort = True
                if early_abort:
                    result = EvalResult(-float('inf'), 0.0, 0.0, 0.0, None)
                    _cache_set(fp, result)
                    return result

            # IC calculation
            return_timestamp_for_ic = common_time_index[t_idx + eval_lag]
            actual_returns_t_slice = np.array([aligned_dfs[sym].loc[return_timestamp_for_ic, "ret_fwd"] for sym in stock_symbols], dtype=float)
            actual_returns_t = np.nan_to_num(actual_returns_t_slice, nan=0.0, posinf=0.0, neginf=0.0) # Clean returns

            if np.all(np.isnan(actual_returns_t)): # If all returns are NaN (e.g. end of data for some reason)
                daily_ic_values.append(0.0)
            else:
                ic_t = _safe_corr_eval(processed_predictions_t, actual_returns_t)
                daily_ic_values.append(0.0 if np.isnan(ic_t) else ic_t)

        except Exception: # Broad exception during program's .eval() or IC calculation
            _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, None))
            return _eval_cache[fp]

    if not daily_ic_values or not all_raw_predictions_timeseries or not all_processed_predictions_timeseries:
        _cache_set(fp, EvalResult(-float('inf'), 0.0, 0.0, 0.0, None))
        return _eval_cache[fp]

    mean_daily_ic = float(np.mean(daily_ic_values))
    parsimony_penalty = _EVAL_CONFIG["parsimony_penalty_factor"] * prog.size / _EVAL_CONFIG["max_ops_for_parsimony"]
    score = mean_daily_ic - parsimony_penalty
    correlation_penalty = 0.0

    full_raw_predictions_matrix = np.array(all_raw_predictions_timeseries)
    full_processed_predictions_matrix = np.array(all_processed_predictions_timeseries)

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
                score = -float('inf')


    # HOF correlation penalty (uses processed predictions)
    if score > -float('inf') and full_processed_predictions_matrix.size > 0:
        correlation_penalty = hof_module.get_correlation_penalty_with_hof(full_processed_predictions_matrix.flatten())
        if correlation_penalty > 0:
            logger.debug(
                "Correlation penalty for %s: %.6f",
                fp,
                correlation_penalty,
            )
        score -= correlation_penalty
    result = EvalResult(score, mean_daily_ic, parsimony_penalty, correlation_penalty, full_processed_predictions_matrix)
    logger.debug(
        "Eval summary %s | fitness %.6f mean_ic %.6f parsimony %.6f correlation %.6f ops %d",
        fp,
        result.fitness,
        result.mean_ic,
        result.parsimony_penalty,
        result.correlation_penalty,
        prog.size,
    )
    _cache_set(fp, result)
    return result
