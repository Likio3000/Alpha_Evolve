from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Dict, Optional, Set # Added Set
import textwrap # For printing HoF

if TYPE_CHECKING:
    from alpha_framework import AlphaProgram

# Module-level state for Hall of Fame
_hof_programs_data: List[Tuple[str, float, float, AlphaProgram, np.ndarray]] = [] # fp, fitness, ic, prog, processed_preds_matrix
_hof_max_size: int = 20
_hof_fingerprints_set: Set[str] = set() # For quick check of existence
_keep_dupes_in_hof_config: bool = False # Corresponds to KEEP_DUPES_IN_HOF_CONFIG

# For correlation penalty
_hof_processed_prediction_timeseries_for_corr: List[np.ndarray] = []
_hof_corr_fingerprints: List[str] = []  # keep order to manage eviction
_corr_penalty_config: Dict[str, float] = {"weight": 0.25, "cutoff": 0.20}


def initialize_hof(max_size: int, keep_dupes: bool, corr_penalty_weight: float, corr_cutoff: float):
    global _hof_programs_data, _hof_max_size, _hof_fingerprints_set, _keep_dupes_in_hof_config
    global _hof_processed_prediction_timeseries_for_corr, _corr_penalty_config, _hof_corr_fingerprints
    
    _hof_programs_data = []
    _hof_max_size = max_size
    _hof_fingerprints_set = set()
    _keep_dupes_in_hof_config = keep_dupes # Though original was hardcoded False
    
    _hof_processed_prediction_timeseries_for_corr = []
    _hof_corr_fingerprints = []
    _corr_penalty_config = {"weight": corr_penalty_weight, "cutoff": corr_cutoff}
    print(f"Hall of Fame initialized: max_size={max_size}, keep_dupes={keep_dupes}, corr_penalty_w={corr_penalty_weight}, corr_cutoff={corr_cutoff}")

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float: # Copied from evolve_alphas, will be used for HOF penalty
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))): return 0.0
    if a.std(ddof=0) < 1e-9 or b.std(ddof=0) < 1e-9: return 0.0
    if len(a) != len(b) or len(a) < 2: return 0.0
    # np.corrcoef can still return NaNs if one array is constant, even if std > 1e-9 due to float precision
    with np.errstate(invalid='ignore'): # Suppress "invalid value encountered in true_divide"
        corr_matrix = np.corrcoef(a, b)
    if np.isnan(corr_matrix[0, 1]):
        return 0.0
    return float(corr_matrix[0, 1])

def get_correlation_penalty_with_hof(current_prog_flat_processed_ts: np.ndarray) -> float:
    penalty = 0.0
    if not _hof_processed_prediction_timeseries_for_corr or current_prog_flat_processed_ts.std(ddof=0) < 1e-9:
        return penalty

    high_corrs = []
    for hof_flat_processed_ts in _hof_processed_prediction_timeseries_for_corr:
        if len(current_prog_flat_processed_ts) != len(hof_flat_processed_ts): continue
        if hof_flat_processed_ts.std(ddof=0) < 1e-9: continue

        corr_with_hof = abs(_safe_corr(current_prog_flat_processed_ts, hof_flat_processed_ts))
        if not np.isnan(corr_with_hof) and corr_with_hof > _corr_penalty_config["cutoff"]:
            high_corrs.append(corr_with_hof)
    
    if high_corrs:
        penalty = _corr_penalty_config["weight"] * float(np.mean(high_corrs))
    return penalty

def add_program_to_hof(
    program: AlphaProgram, 
    fitness: float, 
    mean_ic: float, 
    processed_preds_matrix: Optional[np.ndarray]
):
    """Add ``program`` to the Hall of Fame and update correlation tracking.

    The function keeps the main HOF ordered by fitness and maintains a
    separate list of flattened prediction time series used for correlation
    penalties.  Duplicate fingerprints are ignored in the correlation list and
    the size of both structures is capped at ``_hof_max_size``.
    """
    global _hof_programs_data, _hof_fingerprints_set, _hof_processed_prediction_timeseries_for_corr, _hof_corr_fingerprints

    fp = program.fingerprint
    
    # Logic for adding to _hof_programs_data (main HOF for output)
    # This HOF considers fitness for ranking and uniqueness based on _keep_dupes_in_hof_config
    if not _keep_dupes_in_hof_config and fp in _hof_fingerprints_set:
        # If not keeping dupes and already exists, only update if new one is better
        existing_idx = -1
        for i, (efp, efit, _, _, _) in enumerate(_hof_programs_data):
            if efp == fp:
                existing_idx = i
                break
        if existing_idx != -1 and fitness > _hof_programs_data[existing_idx][1]:
            _hof_programs_data[existing_idx] = (fp, fitness, mean_ic, program, processed_preds_matrix if processed_preds_matrix is not None else np.array([]))
        elif existing_idx == -1: # Should not happen if fp in _hof_fingerprints_set, but as a safeguard
             _hof_programs_data.append((fp, fitness, mean_ic, program, processed_preds_matrix if processed_preds_matrix is not None else np.array([])))
             _hof_fingerprints_set.add(fp)

    else: # Keeping dupes or it's a new fingerprint
        _hof_programs_data.append((fp, fitness, mean_ic, program, processed_preds_matrix if processed_preds_matrix is not None else np.array([])))
        _hof_fingerprints_set.add(fp)

    _hof_programs_data.sort(key=lambda x: x[1], reverse=True) # Sort by fitness
    if len(_hof_programs_data) > _hof_max_size:
        removed_prog_data = _hof_programs_data.pop()
        # If we remove a unique program, ensure its fingerprint is also removed from the set
        # This needs care if multiple entries could share an fp (if _keep_dupes_in_hof_config was true)
        # For now, assuming if _keep_dupes_in_hof_config is false, fingerprints in _hof_programs_data are unique.
        if not any(item[0] == removed_prog_data[0] for item in _hof_programs_data):
            _hof_fingerprints_set.discard(removed_prog_data[0])


    # Logic for maintaining the list used for correlation penalty.
    if processed_preds_matrix is not None and fitness > -float("inf"):
        if fp not in _hof_corr_fingerprints:
            _hof_processed_prediction_timeseries_for_corr.append(processed_preds_matrix.ravel())
            _hof_corr_fingerprints.append(fp)
            if len(_hof_processed_prediction_timeseries_for_corr) > _hof_max_size:
                _hof_processed_prediction_timeseries_for_corr.pop(0)
                _hof_corr_fingerprints.pop(0)


def update_correlation_hof(program_fp: str, processed_preds_matrix: np.ndarray):
    """Add a program's predictions to the correlation HOF, ensuring uniqueness."""
    global _hof_processed_prediction_timeseries_for_corr, _hof_corr_fingerprints

    if program_fp in _hof_corr_fingerprints:
        return

    _hof_processed_prediction_timeseries_for_corr.append(processed_preds_matrix.ravel())
    _hof_corr_fingerprints.append(program_fp)
    if len(_hof_processed_prediction_timeseries_for_corr) > _hof_max_size:
        _hof_processed_prediction_timeseries_for_corr.pop(0)
        _hof_corr_fingerprints.pop(0)


def get_final_hof_programs() -> List[Tuple[AlphaProgram, float]]:
    """Returns the final HOF (program, mean_ic) for output/pickling."""
    # Current HOF stores (fp, fitness, ic, prog, preds_matrix)
    # The original evolve() returned List[Tuple[AlphaProgram, float]] where float was mean_ic.
    
    final_list: List[Tuple[AlphaProgram, float]] = []
    if _keep_dupes_in_hof_config:
        for _, _, ic, prog, _ in _hof_programs_data:
            final_list.append((prog, ic))
            if len(final_list) >= _hof_max_size:
                break
    else:
        # _hof_programs_data is already sorted by fitness and should contain unique FPs if not _keep_dupes
        # (or best version of duplicate FPs)
        unique_output_progs: Dict[str, Tuple[AlphaProgram, float]] = {}
        for fp, _, ic, prog, _ in _hof_programs_data:
            if fp not in unique_output_progs: # Add first encountered (best fitness)
                unique_output_progs[fp] = (prog, ic)
            if len(unique_output_progs) >= _hof_max_size:
                break
        final_list = list(unique_output_progs.values())
        
    return final_list


_TOP_TO_SHOW_PRINT = 10 # From original _update_and_print_hof

def print_generation_summary(generation: int, population: List[AlphaProgram], eval_results_sorted: list): # eval_results_sorted: (idx_in_pop, score, ic, processed_preds_matrix)
    """Prints the HOF summary like _update_and_print_hof."""
    # This function will use the current state of _hof_programs_data
    
    # First, ensure _hof_programs_data is up-to-date with the current generation's best if they qualify
    # This logic was part of the original _update_and_print_hof
    # For each of the top N (e.g., _TOP_TO_SHOW_PRINT) from eval_results_sorted of current gen:
    temp_hof_candidates = []
    for pop_idx, fit, ic, preds_matrix_or_none in eval_results_sorted[:_TOP_TO_SHOW_PRINT]:
        if fit <= -float('inf'): continue # Skip invalid programs
        prog = population[pop_idx]
        fp = prog.fingerprint
        # Add to a temporary list for consideration against the static HOF
        # The actual add_program_to_hof would have been called for the single best or all
        # This print function is more about "displaying" the current top from _hof_programs_data
        temp_hof_candidates.append((fp, fit, ic, prog, preds_matrix_or_none if preds_matrix_or_none is not None else np.array([])))

    # Merge current generation's best with existing static HOF for display purposes
    # This is tricky because add_program_to_hof already updates the static HOF.
    # The printing should just reflect the current state of _hof_programs_data.

    print(f"\n★ Generation {generation+1} – Top (up to) {_TOP_TO_SHOW_PRINT} overall from HOF ★")
    hdr = " Rank | Fitness |  IC  | Ops | Finger  | First 90 chars"
    print(hdr)
    print("─" * len(hdr))
    
    # Print from the managed _hof_programs_data
    for rk, (fp, fit, ic, prog, _) in enumerate(_hof_programs_data[:_TOP_TO_SHOW_PRINT], 1):
        head = textwrap.shorten(prog.to_string(max_len=300), width=90, placeholder="…")
        print(f" {rk:>4} | {fit:+7.4f} | {ic:+5.3f} | {prog.size:3d} | {fp[:8]} | {head}")
    print()


def clear_hof():
    """Clears all HOF state."""
    global _hof_programs_data, _hof_fingerprints_set, _hof_processed_prediction_timeseries_for_corr, _hof_corr_fingerprints
    _hof_programs_data = []
    _hof_fingerprints_set = set()
    _hof_processed_prediction_timeseries_for_corr = []
    _hof_corr_fingerprints = []
    print("Hall of Fame cleared.")
