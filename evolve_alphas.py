from __future__ import annotations
"""
evolve_alphas.py – evolution driver
"""

from pathlib import Path
import argparse
import os
import glob
import random
import sys
import time
from typing import Dict, List, Tuple, Optional, Any, OrderedDict as OrderedDictType, Set
from collections import OrderedDict
import textwrap
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


from alpha_program_core import (
    AlphaProgram, Op, TypeId,
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    SCALAR_FEATURE_NAMES,
    FINAL_PREDICTION_VECTOR_NAME,
    OP_REGISTRY
)
import pickle




###############################################################################
# CLI & CONFIG ################################################################
###############################################################################

def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cross-Sectional Evolutionary Alpha Search")
    ap.add_argument("generations", nargs="?", type=int, default=10)
    ap.add_argument("seed", nargs="?", type=int, default=42)
    ap.add_argument("-q", "--quiet", action="store_true", help="hide progress bar")
    ap.add_argument("--max_lookback_data_option", type=str, choices=['common_1200', 'specific_long_10k', 'full_overlap'], default='common_1200')
    ap.add_argument("--min_common_points", type=int, default=1200)
    ap.add_argument("--data_dir", default="./data", help="Directory with *.csv OHLC data")
    ap.add_argument("--pop_size", type=int, default=64)
    ap.add_argument("--tournament_k", type=int, default=4)
    ap.add_argument("--p_mut", type=float, default=0.4)
    ap.add_argument("--p_cross", type=float, default=0.6)
    ap.add_argument("--elite_keep", type=int, default=4)
    ap.add_argument("--max_ops", type=int, default=32)
    ap.add_argument("--parsimony_penalty", type=float, default=0.002)
    ap.add_argument("--corr_penalty_w", type=float, default=0.25)
    ap.add_argument("--corr_cutoff", type=float, default=0.20)
    ap.add_argument("--hof_size", type=int, default=20)
    ap.add_argument("--scale", default="zscore", choices=["zscore","rank","sign"])
    ap.add_argument("--eval_lag", type=int, default=1)
    ap.add_argument("--fresh_rate", type=float, default=0.05, help="Probability that a slot in the new population is filled by a completely random program (novelty injection). Set to 0 to disable.")
    return ap.parse_args()

### Upsie
if __name__ == "__main__" or "pytest" in sys.modules:
    args = _parse_cli()

else:
    # module is being imported by run_pipeline.py
    args = argparse.Namespace(
        generations=1, seed=0, quiet=False,
        max_lookback_data_option="common_1200", min_common_points=1200,
        data_dir="./data", pop_size=1, tournament_k=2,
        p_mut=0.0, p_cross=0.0, elite_keep=0, max_ops=1,
        parsimony_penalty=0.0, corr_penalty_w=0.0, corr_cutoff=1.0,
        hof_size=0, scale="zscore", eval_lag=1,
        fresh_rate=0.05
    )


DATA_DIR = args.data_dir
POP_SIZE = args.pop_size
N_GENERATIONS = args.generations
TOURNAMENT_K = args.tournament_k
P_MUT = args.p_mut
P_CROSS = args.p_cross
ELITE_KEEP = args.elite_keep
MAX_OPS = args.max_ops
PARSIMONY_PENALTY = args.parsimony_penalty
CORR_PENALTY_W = args.corr_penalty_w
CORR_CUTOFF = args.corr_cutoff
DUPLICATE_HOF_SZ = args.hof_size
SEED = args.seed
EVAL_LAG = args.eval_lag
FRESH_RATE = args.fresh_rate

XS_FLATNESS_GUARD_THRESHOLD = 5e-2
EARLY_ABORT_BARS = 20
EARLY_ABORT_XS_THRESHOLD = 5e-2
EARLY_ABORT_T_THRESHOLD = 5e-2
KEEP_DUPES_IN_HOF_CONFIG = False


random.seed(SEED)
np.random.seed(SEED)

###############################################################################
# 1. DATA LOADING & PREPARATION ###############################################
###############################################################################
FEATURE_VARS: Dict[str, TypeId] = {name: "vector" for name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES}
FEATURE_VARS.update({name: "scalar" for name in SCALAR_FEATURE_NAMES})
if "const_1" not in FEATURE_VARS: FEATURE_VARS["const_1"] = "scalar"
if "const_neg_1" not in FEATURE_VARS: FEATURE_VARS["const_neg_1"] = "scalar"

INITIAL_STATE_VARS: Dict[str, TypeId] = {
    "prev_s1_vec": "vector",
    "rolling_mean_custom": "vector"
}

_ALIGNED_DFS: Optional[OrderedDictType[str, pd.DataFrame]] = None
_COMMON_TIME_INDEX: Optional[pd.DatetimeIndex] = None
_STOCK_SYMBOLS: Optional[List[str]] = None
_N_STOCKS: Optional[int] = None

def _sync_constants_from_args():
    global DATA_DIR, POP_SIZE, N_GENERATIONS, TOURNAMENT_K, P_MUT, P_CROSS, ELITE_KEEP
    global MAX_OPS, PARSIMONY_PENALTY, CORR_PENALTY_W, CORR_CUTOFF, DUPLICATE_HOF_SZ, SEED, EVAL_LAG, FRESH_RATE

    DATA_DIR = args.data_dir
    POP_SIZE = args.pop_size
    N_GENERATIONS = args.generations
    TOURNAMENT_K = args.tournament_k
    P_MUT = args.p_mut
    P_CROSS = args.p_cross
    ELITE_KEEP = args.elite_keep
    MAX_OPS = args.max_ops
    PARSIMONY_PENALTY = args.parsimony_penalty
    CORR_PENALTY_W = args.corr_penalty_w
    CORR_CUTOFF = args.corr_cutoff
    DUPLICATE_HOF_SZ = args.hof_size
    EVAL_LAG = args.eval_lag
    FRESH_RATE = args.fresh_rate
    if SEED != args.seed:
        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)


def _ensure_data_loaded():
    global _ALIGNED_DFS, _COMMON_TIME_INDEX, _STOCK_SYMBOLS, _N_STOCKS, args

    if _ALIGNED_DFS is not None:
        return

    _sync_constants_from_args()

    df_data, time_idx_data, symbols_data = load_and_align_data(
        args.data_dir, args.max_lookback_data_option, args.min_common_points
    )
    _ALIGNED_DFS = df_data
    _COMMON_TIME_INDEX = time_idx_data
    _STOCK_SYMBOLS = symbols_data
    _N_STOCKS = len(_STOCK_SYMBOLS)

    print(f"evolve_alphas: Using {_N_STOCKS} symbols for evolution ")
    print(f"Data spans {_COMMON_TIME_INDEX.min()} to {_COMMON_TIME_INDEX.max()} with {_COMMON_TIME_INDEX.size} steps. Eval lag: {EVAL_LAG}\n")

def _rolling_features_individual_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in (5, 10, 20, 30):
        df[f"ma{w}"] = df["close"].rolling(w, min_periods=1).mean()
        df[f"vol{w}"] = df["close"].rolling(w, min_periods=1).std(ddof=0)
    df["range"] = df["high"] - df["low"]
    df["ret_fwd"] = df["close"].pct_change(periods=1).shift(-1)
    return df

def load_and_align_data(data_dir_param: str, strategy_param: str, min_common_points_param: int) -> Tuple[OrderedDictType[str, pd.DataFrame], pd.DatetimeIndex, List[str]]:
    raw_dfs: Dict[str, pd.DataFrame] = {}
    for csv_file in glob.glob(os.path.join(data_dir_param, "*.csv")):
        try:
            df = pd.read_csv(csv_file)
            if 'time' not in df.columns:
                continue
            df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
            df = df.dropna(subset=['time']).sort_values("time").set_index("time")
            if df.empty: continue
            df_with_features = _rolling_features_individual_df(df)
            raw_dfs[Path(csv_file).stem] = df_with_features.dropna()
        except Exception:
            continue

    if not raw_dfs:
        sys.exit(f"No valid CSV data loaded from {data_dir_param}.")

    if strategy_param == 'specific_long_10k':
        min_len_for_long = min_common_points_param
        raw_dfs = {sym: df for sym, df in raw_dfs.items() if len(df) >= min_len_for_long}
        if len(raw_dfs) < 2:
             sys.exit(f"Not enough long files (>= {min_len_for_long} data points) found for 'specific_long_10k' strategy. Found: {len(raw_dfs)}")

    common_index: Optional[pd.DatetimeIndex] = None
    for df_sym in raw_dfs.values():
        if common_index is None: common_index = df_sym.index
        else: common_index = common_index.intersection(df_sym.index)

    required_length_for_eval = min_common_points_param + EVAL_LAG
    if common_index is None or len(common_index) < required_length_for_eval:
        sys.exit(f"Not enough common history (need {required_length_for_eval} for {min_common_points_param} eval steps + lag {EVAL_LAG}, got {len(common_index if common_index is not None else [])}).")

    if strategy_param == 'common_1200' or strategy_param == 'specific_long_10k':
        num_points_to_keep = min_common_points_param + EVAL_LAG
        if len(common_index) > num_points_to_keep:
            common_index = common_index[-num_points_to_keep:]

    aligned_dfs_ordered = OrderedDict()
    for sym in sorted(raw_dfs.keys()):
        df_sym = raw_dfs[sym]
        reindexed_df = df_sym.reindex(common_index).ffill().bfill()
        if reindexed_df.isnull().values.any():
             print(f"Warning: DataFrame for {sym} still contains NaNs after ffill/bfill on common_index.")
        aligned_dfs_ordered[sym] = reindexed_df

    stock_symbols = list(aligned_dfs_ordered.keys())
    if len(stock_symbols) < 2:
        sys.exit("Need at least two stock symbols after alignment for cross-sectional evolution.")
    return aligned_dfs_ordered, common_index, stock_symbols

###############################################################################
# 2. SAFE CORRELATION + CACHES + GUARDS #######################################
###############################################################################
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))): return 0.0
    if a.std(ddof=0) < 1e-9 or b.std(ddof=0) < 1e-9: return 0.0
    if len(a) != len(b) or len(a) < 2: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

_HOF_fingerprints: List[str] = []
_HOF_processed_prediction_timeseries: List[np.ndarray] = [] # MODIFIED: Store processed predictions for HOF
_eval_cache: Dict[str, Tuple[float, float, Optional[np.ndarray]]] = {}


def _uses_feature_vector(prog: AlphaProgram) -> bool:
    all_ops_map: Dict[str, Op] = {}
    for op_instance in prog.setup + prog.predict_ops + prog.update_ops:
        all_ops_map[op_instance.out] = op_instance

    if FINAL_PREDICTION_VECTOR_NAME not in all_ops_map and \
       FINAL_PREDICTION_VECTOR_NAME not in INITIAL_STATE_VARS and \
       FINAL_PREDICTION_VECTOR_NAME not in FEATURE_VARS:
        return FINAL_PREDICTION_VECTOR_NAME in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES

    q: List[str] = [FINAL_PREDICTION_VECTOR_NAME]
    visited_vars: Set[str] = set()

    while q:
        current_var_name = q.pop()
        if current_var_name in visited_vars: continue
        visited_vars.add(current_var_name)

        if current_var_name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
            return True

        defining_op = all_ops_map.get(current_var_name)
        if defining_op:
            for input_var_name in defining_op.inputs:
                if input_var_name not in visited_vars:
                    q.append(input_var_name)
    return False


###############################################################################
# 3. PROGRAM EXECUTION & EVALUATION ###########################################
###############################################################################

def _scale_signal_cross_sectionally_for_ic(raw_signal_vector: np.ndarray, method: str) -> np.ndarray:
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
    else:
        mu = np.nanmean(clean_signal_vector)
        sd = np.nanstd(clean_signal_vector)
        if sd < 1e-9 :
            scaled = np.zeros_like(clean_signal_vector)
        else:
            scaled = (clean_signal_vector - mu) / sd
    
    # Centering after scaling for IC calculation and HOF comparison
    centered_scaled = scaled - np.mean(scaled)
    return np.clip(centered_scaled, -1, 1) # Clip after centering


def evaluate(prog: AlphaProgram) -> Tuple[float, float, Optional[np.ndarray]]:
    fp = prog.fingerprint
    if fp in _eval_cache: return _eval_cache[fp]

    if not _uses_feature_vector(prog):
        _eval_cache[fp] = (-float('inf'), 0.0, None)
        return _eval_cache[fp]

    program_state: Dict[str, Any] = prog.new_state()
    for s_name, s_type in INITIAL_STATE_VARS.items():
        if s_name not in program_state:
            if s_type == "vector": program_state[s_name] = np.zeros(_N_STOCKS)
            elif s_type == "matrix": program_state[s_name] = np.zeros((_N_STOCKS, _N_STOCKS))
            else: program_state[s_name] = 0.0

    all_raw_predictions_timeseries: List[np.ndarray] = [] # Store raw for flatness guards
    all_processed_predictions_timeseries: List[np.ndarray] = [] # MODIFIED: Store processed for HOF and IC
    daily_ic_values: List[float] = []

    loop_end_idx = len(_COMMON_TIME_INDEX) - EVAL_LAG -1
    if loop_end_idx < 0:
        _eval_cache[fp] = (-float('inf'), 0.0, None)
        return _eval_cache[fp]

    for t_idx in range(loop_end_idx +1):
        timestamp = _COMMON_TIME_INDEX[t_idx]

        features_at_t: Dict[str, Any] = {}
        for feat_name_template in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
            col_name = feat_name_template.replace('_t', '')
            try:
                feat_vec = np.array([_ALIGNED_DFS[sym].loc[timestamp, col_name] for sym in _STOCK_SYMBOLS], dtype=float)
                features_at_t[feat_name_template] = np.nan_to_num(feat_vec, nan=0.0)
            except KeyError:
                features_at_t[feat_name_template] = np.zeros(_N_STOCKS, dtype=float)

        for sc_name in SCALAR_FEATURE_NAMES:
            if sc_name == "const_1": features_at_t[sc_name] = 1.0
            elif sc_name == "const_neg_1": features_at_t[sc_name] = -1.0

        try:
            raw_predictions_t = prog.eval(features_at_t, program_state, _N_STOCKS)
            if np.any(np.isnan(raw_predictions_t)) or np.any(np.isinf(raw_predictions_t)):
                _eval_cache[fp] = (-float('inf'), 0.0, None)
                return _eval_cache[fp]
            
            all_raw_predictions_timeseries.append(raw_predictions_t.copy())
            
            # MODIFIED: Scale and center predictions *before* HOF comparison and IC calculation
            processed_predictions_t = _scale_signal_cross_sectionally_for_ic(raw_predictions_t, args.scale)
            all_processed_predictions_timeseries.append(processed_predictions_t)


            if len(all_raw_predictions_timeseries) >= EARLY_ABORT_BARS:
                if len(all_raw_predictions_timeseries) == EARLY_ABORT_BARS:
                    partial_raw_preds_matrix = np.array(all_raw_predictions_timeseries)
                    mean_xs_std_partial = 0.0
                    if partial_raw_preds_matrix.ndim == 2 and partial_raw_preds_matrix.shape[1] > 0:
                        mean_xs_std_partial = np.mean(partial_raw_preds_matrix.std(axis=1, ddof=0))
                    mean_t_std_partial = 0.0
                    if partial_raw_preds_matrix.ndim == 2 and partial_raw_preds_matrix.shape[0] > 1:
                        mean_t_std_partial = np.mean(partial_raw_preds_matrix.std(axis=0, ddof=0))
                    elif partial_raw_preds_matrix.ndim == 1 and partial_raw_preds_matrix.shape[0] > 1:
                        mean_t_std_partial = partial_raw_preds_matrix.std(ddof=0)

                    if mean_xs_std_partial < EARLY_ABORT_XS_THRESHOLD or \
                       mean_t_std_partial < EARLY_ABORT_T_THRESHOLD:
                        _eval_cache[fp] = (-float('inf'), 0.0, None)
                        return _eval_cache[fp]

            return_timestamp_for_ic = _COMMON_TIME_INDEX[t_idx + EVAL_LAG]
            actual_returns_t = np.array([_ALIGNED_DFS[sym].loc[return_timestamp_for_ic, "ret_fwd"] for sym in _STOCK_SYMBOLS], dtype=float)

            if np.any(np.isnan(actual_returns_t)):
                daily_ic_values.append(0.0)
            else:
                # Use the already scaled and centered predictions for IC
                ic_t = _safe_corr(processed_predictions_t, actual_returns_t)
                daily_ic_values.append(0.0 if np.isnan(ic_t) else ic_t)

        except Exception:
            _eval_cache[fp] = (-float('inf'), 0.0, None)
            return _eval_cache[fp]

    if not daily_ic_values or not all_raw_predictions_timeseries or not all_processed_predictions_timeseries:
        _eval_cache[fp] = (-float('inf'), 0.0, None)
        return _eval_cache[fp]

    mean_daily_ic = float(np.mean(daily_ic_values))
    score = mean_daily_ic - PARSIMONY_PENALTY * prog.size / MAX_OPS

    full_raw_predictions_matrix = np.array(all_raw_predictions_timeseries)
    full_processed_predictions_matrix = np.array(all_processed_predictions_timeseries) # MODIFIED

    # Cross-sectional flatness guard (on raw predictions)
    if full_raw_predictions_matrix.ndim == 2 and full_raw_predictions_matrix.shape[1] > 0:
        cross_sectional_stds = full_raw_predictions_matrix.std(axis=1, ddof=0)
        if np.mean(cross_sectional_stds) < XS_FLATNESS_GUARD_THRESHOLD:
            score = -float('inf')
            _eval_cache[fp] = (score, mean_daily_ic, full_processed_predictions_matrix) # Cache processed for HOF
            return score, mean_daily_ic, full_processed_predictions_matrix

    # Temporal flatness guard (on raw predictions)
    flat_signal_threshold = 5e-2
    if full_raw_predictions_matrix.ndim == 2 and full_raw_predictions_matrix.shape[0] > 1:
        time_std_per_stock = full_raw_predictions_matrix.std(axis=0, ddof=0)
        if np.mean(time_std_per_stock) < flat_signal_threshold:
            score = -float('inf')
    elif full_raw_predictions_matrix.ndim == 1 and full_raw_predictions_matrix.shape[0] > 1:
        if full_raw_predictions_matrix.std(ddof=0) < flat_signal_threshold:
            score = -float('inf')

    # MODIFIED: HOF correlation penalty using flattened PROCESSED predictions
    if _HOF_processed_prediction_timeseries and score > -float('inf'):
        current_prog_flat_processed_ts = full_processed_predictions_matrix.flatten()
        high_corrs = []
        for hof_processed_preds_matrix in _HOF_processed_prediction_timeseries:
            hof_flat_processed_ts = hof_processed_preds_matrix.flatten()

            if len(current_prog_flat_processed_ts) != len(hof_flat_processed_ts): continue
            if current_prog_flat_processed_ts.std(ddof=0) < 1e-9 or hof_flat_processed_ts.std(ddof=0) < 1e-9: continue

            corr_with_hof = abs(_safe_corr(current_prog_flat_processed_ts, hof_flat_processed_ts))
            if not np.isnan(corr_with_hof) and corr_with_hof > CORR_CUTOFF:
                high_corrs.append(corr_with_hof)
        if high_corrs:
            score -= CORR_PENALTY_W * float(np.mean(high_corrs))

    _eval_cache[fp] = (score, mean_daily_ic, full_processed_predictions_matrix) # Store processed matrix
    return score, mean_daily_ic, full_processed_predictions_matrix


###############################################################################
# 4. EA OPERATORS (Wrappers) ##################################################
###############################################################################
def _random_prog() -> AlphaProgram:
    return AlphaProgram.random_program(FEATURE_VARS, INITIAL_STATE_VARS, max_total_ops=MAX_OPS)

def _mutate_prog(p: AlphaProgram) -> AlphaProgram:
    return p.mutate(FEATURE_VARS, INITIAL_STATE_VARS, max_total_ops=MAX_OPS)

def _crossover_prog(a: AlphaProgram, b: AlphaProgram) -> AlphaProgram:
    return a.crossover(b)

###############################################################################
# 5. PROGRESS BAR WRAPPER #####################################################
###############################################################################
def _pbar(iterable, *, desc: str, disable: bool):
    if tqdm and not disable:
        return tqdm(iterable, desc=desc, leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    class _DummyPBar:
        def __init__(self, it, **kwargs): self._it = iter(it)
        def __iter__(self): return self
        def __next__(self): return next(self._it)
        def update(self, *_args, **_kwargs): pass
        def close(self): pass
        def set_postfix_str(self, s): pass
    return _DummyPBar(iterable)

###############################################################################
# 6. EVOLVE LOOP ##############################################################
###############################################################################

# ──────────────────────────────────────────────────────────────
# Rolling Hall-of-Fame printer
# ──────────────────────────────────────────────────────────────
_TOP_TO_SHOW = 10          # how many elites you want to track & print
_static_hof: list[tuple[str, float, float, AlphaProgram]] = []

def _update_and_print_hof(pop, eval_results, generation):
    """
    Merge this generation’s best candidates into _static_hof,
    keep the global top N by fitness, and print a nice table.
    """
    global _static_hof
    seen = {fp for fp, *_ in _static_hof}

    # 1⃣  merge newcomers
    for idx, fit, ic, _ in eval_results[:_TOP_TO_SHOW]:
        prog = pop[idx]
        fp = prog.fingerprint
        if fp not in seen:
            _static_hof.append((fp, fit, ic, prog))
            seen.add(fp)

    # 2⃣  keep only the best N by fitness
    _static_hof = sorted(_static_hof, key=lambda t: t[1], reverse=True)[:_TOP_TO_SHOW]

    # 3⃣  pretty-print
    print(f"\n★ Generation {generation+1} – Top {_TOP_TO_SHOW} overall ★")
    hdr = " Rank | Fitness |  IC  | Ops | Finger  | First 90 chars"
    print(hdr)
    print("─" * len(hdr))
    for rk, (fp, fit, ic, prog) in enumerate(_static_hof, 1):
        head = textwrap.shorten(prog.to_string(max_len=300), width=90, placeholder="…")
        print(f" {rk:>4} | {fit:+7.4f} | {ic:+5.3f} | {prog.size:3d} | {fp[:8]} | {head}")
    print()        # blank line for readability


def evolve() -> List[Tuple[AlphaProgram, float]]:
    _ensure_data_loaded()

    pop: List[AlphaProgram] = [_random_prog() for _ in range(POP_SIZE)]
    global _HOF_processed_prediction_timeseries, _HOF_fingerprints, _eval_cache # MODIFIED HOF variable name
    _HOF_processed_prediction_timeseries = [] # MODIFIED
    _HOF_fingerprints = []
    _eval_cache = {}
    gen_eval_times_history: List[float] = []

    try:
        for gen in range(N_GENERATIONS):
            t_start_gen = time.perf_counter()
            eval_results: List[Tuple[int, float, float, Optional[np.ndarray]]] = []
            pop_fitness_scores = np.full(POP_SIZE, -np.inf)

            bar = _pbar(range(POP_SIZE), desc=f"Gen {gen+1}/{N_GENERATIONS}", disable=args.quiet)
            for i in bar:
                prog = pop[i]
                score, mean_ic, processed_preds_matrix = evaluate(prog) # 'processed_preds_matrix' used for HOF
                eval_results.append((i, score, mean_ic, processed_preds_matrix))
                pop_fitness_scores[i] = score
                if tqdm and not args.quiet:
                    valid_scores = pop_fitness_scores[pop_fitness_scores > -np.inf]
                    if valid_scores.size > 0:
                        bar.set_postfix_str(f"BestFit: {np.max(valid_scores):.4f}")
                    else:
                        bar.set_postfix_str(f"BestFit: N/A")

            gen_eval_time = time.perf_counter() - t_start_gen
            if gen_eval_time > 0:
                gen_eval_times_history.append(gen_eval_time)
            eval_results.sort(key=lambda x: x[1], reverse=True)
            _update_and_print_hof(pop, eval_results, gen)

            if not eval_results or eval_results[0][1] <= -float('inf'):
                print(f"Gen {gen+1:3d} | No valid programs. Restarting population.")
                pop = [_random_prog() for _ in range(POP_SIZE)]
                _eval_cache.clear()
                _HOF_fingerprints.clear()
                _HOF_processed_prediction_timeseries.clear() # MODIFIED
                gen_eval_times_history.clear()
                continue

            eta_str = ""
            if gen_eval_times_history:
                avg_gen_time = np.mean(gen_eval_times_history)
                remaining_gens = N_GENERATIONS - (gen + 1)
                if avg_gen_time > 0 and remaining_gens > 0:
                    eta_seconds = remaining_gens * avg_gen_time
                    eta_str = f" | ETA {time.strftime('%Hh%Mm%Ss', time.gmtime(eta_seconds))}"

            best_prog_idx_in_pop, best_fit, best_ic, best_processed_preds_matrix = eval_results[0]
            best_program_this_gen = pop[best_prog_idx_in_pop]
            tail_alpha = best_program_this_gen.to_string(max_len=1000000)

            print(
                f"Gen {gen+1:3d} \n"
                f"BestFit {best_fit:+.4f} \n"
                f"MeanIC {best_ic:+.4f} \n"
                f"Ops {best_program_this_gen.size:2d} \n"
                f"EvalTime {gen_eval_time:.1f}s{eta_str} \n"
                f"{tail_alpha[-100:]}"
            )

            # MODIFIED: Use best_processed_preds_matrix for HOF
            if best_processed_preds_matrix is not None and best_fit > -float('inf'):
                fp_best = best_program_this_gen.fingerprint
                if fp_best not in _HOF_fingerprints:
                    _HOF_fingerprints.append(fp_best)
                    _HOF_processed_prediction_timeseries.append(best_processed_preds_matrix) # MODIFIED

                    if len(_HOF_fingerprints) > DUPLICATE_HOF_SZ:
                        _HOF_fingerprints.pop(0)
                        _HOF_processed_prediction_timeseries.pop(0) # MODIFIED

            new_pop: List[AlphaProgram] = []
            elites_added_fingerprints = set()
            for i_res, score_res, ic_res, _ in eval_results:
                if score_res <= -float('inf'): continue
                prog_candidate = pop[i_res]
                fp_cand = prog_candidate.fingerprint
                if fp_cand not in elites_added_fingerprints:
                    new_pop.append(prog_candidate.copy())
                    elites_added_fingerprints.add(fp_cand)
                if len(new_pop) >= ELITE_KEEP:
                    break

            if not new_pop and eval_results and eval_results[0][1] > -float('inf'):
                 new_pop.append(pop[eval_results[0][0]].copy())


            while len(new_pop) < POP_SIZE:
                # ❶  FRESH / NOVELTY INJECTION
                # ─────────────────────────────────────────────────────────────
                if random.random() < FRESH_RATE:            #  ⚑
                    new_pop.append(_random_prog())          #     drop a random program in
                    continue                                #     and move straight to the next slot
                valid_indices = [k_idx for k_idx, s_k in enumerate(pop_fitness_scores) if s_k > -np.inf]

                if not valid_indices:
                    new_pop.extend([_random_prog() for _ in range(POP_SIZE - len(new_pop))])
                    break

                num_to_sample = TOURNAMENT_K * 2
                if len(valid_indices) < num_to_sample :
                    tournament_indices_pool = random.choices(valid_indices, k=num_to_sample)
                else:
                    tournament_indices_pool = random.sample(valid_indices, num_to_sample)

                parent1_idx = max(tournament_indices_pool[:TOURNAMENT_K], key=lambda i_tour: pop_fitness_scores[i_tour])
                parent2_idx = max(tournament_indices_pool[TOURNAMENT_K:], key=lambda i_tour: pop_fitness_scores[i_tour])

                parent_a, parent_b = pop[parent1_idx], pop[parent2_idx]

                child: AlphaProgram
                if random.random() < P_CROSS: child = _crossover_prog(parent_a, parent_b)
                else: child = parent_a.copy() if random.random() < 0.5 else parent_b.copy()
                if random.random() < P_MUT: child = _mutate_prog(child)
                new_pop.append(child)
            pop = new_pop
    except KeyboardInterrupt:
        print("\n[Ctrl‑C] Evolution stopped early. Processing current generation...")

    final_eval_results: List[Tuple[AlphaProgram, float, float]] = []
    processed_fps: set[str] = set()
    print("\nEvaluating final population for top unique programs...")
    bar = _pbar(pop, desc="Final Eval", disable=args.quiet)
    for prog_final in bar:
        fp_final = prog_final.fingerprint
        if fp_final in processed_fps: continue
        score_final, ic_final, _ = evaluate(prog_final)
        if score_final > -float('inf'):
            final_eval_results.append((prog_final, score_final, ic_final))
        processed_fps.add(fp_final)

    final_eval_results.sort(key=lambda x: x[1], reverse=True)

    if KEEP_DUPES_IN_HOF_CONFIG: # This global toggle is set to False
        top_programs_with_ic_temp = []
        for prog_cand, score_cand, ic_cand in final_eval_results:
            if score_cand > -float('inf'):
                top_programs_with_ic_temp.append((prog_cand, ic_cand))
            if len(top_programs_with_ic_temp) >= DUPLICATE_HOF_SZ:
                break
        top_programs_with_ic = top_programs_with_ic_temp
    else:
        unique_progs_dict: Dict[str, Tuple[AlphaProgram, float]] = {}
        for prog_candidate, score_candidate, ic_candidate in final_eval_results:
            if score_candidate <= -float('inf'):
                continue

            fp_candidate = prog_candidate.fingerprint
            if fp_candidate not in unique_progs_dict:
                unique_progs_dict[fp_candidate] = (prog_candidate, ic_candidate)

            if len(unique_progs_dict) >= DUPLICATE_HOF_SZ:
                break
        top_programs_with_ic = list(unique_progs_dict.values())

    return top_programs_with_ic
