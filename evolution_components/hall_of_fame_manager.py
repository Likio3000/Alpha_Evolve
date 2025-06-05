from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Dict, Optional, Set # Added Set
import textwrap # For printing HoF

if TYPE_CHECKING:
    from alpha_framework import AlphaProgram
    from .evaluation_logic import EvalResult

# Module-level state for Hall of Fame
_hof_programs_data: List[Tuple[str, EvalResult, AlphaProgram]] = []  # fp, metrics, prog
_hof_max_size: int = 20
_hof_fingerprints_set: Set[str] = set() # For quick check of existence
_keep_dupes_in_hof_config: bool = False # Corresponds to KEEP_DUPES_IN_HOF_CONFIG

# For correlation penalty
# Store rank-transformed prediction vectors for correlation checks
_hof_rank_pred_matrix: List[np.ndarray] = []
_hof_corr_fingerprints: List[str] = []  # keep order to manage eviction
# Default correlation penalty configuration mirrors Section 9
_corr_penalty_config: Dict[str, float] = {"weight": 0.35, "cutoff": 0.15}


def initialize_hof(max_size: int, keep_dupes: bool, corr_penalty_weight: float, corr_cutoff: float):
    global _hof_programs_data, _hof_max_size, _hof_fingerprints_set, _keep_dupes_in_hof_config
    global _hof_rank_pred_matrix, _corr_penalty_config, _hof_corr_fingerprints
    
    _hof_programs_data = []
    _hof_max_size = max_size
    _hof_fingerprints_set = set()
    _keep_dupes_in_hof_config = keep_dupes # Though original was hardcoded False
    
    _hof_rank_pred_matrix = []
    _hof_corr_fingerprints = []
    _corr_penalty_config = {"weight": corr_penalty_weight, "cutoff": corr_cutoff}
    print(f"Hall of Fame initialized: max_size={max_size}, keep_dupes={keep_dupes}, corr_penalty_w={corr_penalty_weight}, corr_cutoff={corr_cutoff}")

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:  # Copied from evolve_alphas, will be used for HOF penalty
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0
    if a.std(ddof=0) < 1e-9 or b.std(ddof=0) < 1e-9:
        return 0.0
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    # np.corrcoef can still return NaNs if one array is constant, even if std > 1e-9 due to float precision
    with np.errstate(invalid='ignore'): # Suppress "invalid value encountered in true_divide"
        corr_matrix = np.corrcoef(a, b)
    if np.isnan(corr_matrix[0, 1]):
        return 0.0
    return float(corr_matrix[0, 1])

def _rank_vector(vec: np.ndarray) -> np.ndarray:
    """Return zero-mean ranks for ``vec`` used in Spearman correlation."""
    order = np.argsort(vec)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(vec), dtype=float)
    ranks -= ranks.mean()
    return ranks


def get_correlation_penalty_with_hof(current_prog_flat_processed_ts: np.ndarray) -> float:
    if not _hof_rank_pred_matrix or current_prog_flat_processed_ts.std(ddof=0) < 1e-9:
        return 0.0

    cand_rank = _rank_vector(current_prog_flat_processed_ts)
    corrs: List[float] = []
    for hof_rank in _hof_rank_pred_matrix:
        if len(hof_rank) != len(cand_rank):
            continue
        corr = abs(_safe_corr(cand_rank, hof_rank))
        if not np.isnan(corr) and corr > _corr_penalty_config["cutoff"]:
            corrs.append(corr)

    if not corrs:
        return 0.0
    return _corr_penalty_config["weight"] * float(np.mean(corrs))

def add_program_to_hof(
    program: AlphaProgram,
    metrics: EvalResult,
):
    """Add ``program`` to the Hall of Fame and update correlation tracking.

    The function keeps the main HOF ordered by fitness and maintains a
    separate list of flattened prediction time series used for correlation
    penalties.  Duplicate fingerprints are ignored in the correlation list and
    the size of both structures is capped at ``_hof_max_size``.
    """
    global _hof_programs_data, _hof_fingerprints_set, _hof_rank_pred_matrix, _hof_corr_fingerprints

    fp = program.fingerprint

    # Reject if the new program's predictions are highly correlated with any
    # existing HOF entry.  Skip comparison against entries that share the same
    processed_preds_matrix = metrics.processed_predictions
    # fingerprint (possible updates of an existing program).
    if (
        processed_preds_matrix is not None
        and _hof_rank_pred_matrix
        and processed_preds_matrix.size > 0
    ):
        cand_rank = _rank_vector(processed_preds_matrix.ravel())
        for hof_rank, hof_fp in zip(_hof_rank_pred_matrix, _hof_corr_fingerprints):
            if hof_fp == fp or len(hof_rank) != len(cand_rank):
                continue
            corr = abs(_safe_corr(cand_rank, hof_rank))
            if not np.isnan(corr) and corr > _corr_penalty_config["cutoff"]:
                return  # Too correlated – do not add to the HOF
    
    # Logic for adding to _hof_programs_data (main HOF for output)
    if not _keep_dupes_in_hof_config and fp in _hof_fingerprints_set:
        existing_idx = -1
        for i, (efp, _, _) in enumerate(_hof_programs_data):
            if efp == fp:
                existing_idx = i
                break
        if existing_idx != -1 and metrics.fitness > _hof_programs_data[existing_idx][1].fitness:
            _hof_programs_data[existing_idx] = (fp, metrics, program)
        elif existing_idx == -1:
            _hof_programs_data.append((fp, metrics, program))
            _hof_fingerprints_set.add(fp)
    else:
        _hof_programs_data.append((fp, metrics, program))
        _hof_fingerprints_set.add(fp)


        _hof_programs_data.sort(key=lambda x: x[1].fitness, reverse=True)  # Sort by fitness
    if len(_hof_programs_data) > _hof_max_size:
        removed_prog_data = _hof_programs_data.pop()
        # If we remove a unique program, ensure its fingerprint is also removed from the set
        # This needs care if multiple entries could share an fp (if _keep_dupes_in_hof_config was true)
        # For now, assuming if _keep_dupes_in_hof_config is false, fingerprints in _hof_programs_data are unique.
        if not any(item[0] == removed_prog_data[0] for item in _hof_programs_data):
            _hof_fingerprints_set.discard(removed_prog_data[0])


    # Logic for maintaining the list used for correlation penalty.
    if processed_preds_matrix is not None and metrics.fitness > -float("inf"):
        if fp not in _hof_corr_fingerprints:
            _hof_rank_pred_matrix.append(_rank_vector(processed_preds_matrix.ravel()))
            _hof_corr_fingerprints.append(fp)
            if len(_hof_rank_pred_matrix) > _hof_max_size:
                _hof_rank_pred_matrix.pop(0)
                _hof_corr_fingerprints.pop(0)


def update_correlation_hof(program_fp: str, processed_preds_matrix: np.ndarray):
    """Add a program's predictions to the correlation HOF, ensuring uniqueness."""
    global _hof_rank_pred_matrix, _hof_corr_fingerprints

    if program_fp in _hof_corr_fingerprints:
        return

    _hof_rank_pred_matrix.append(_rank_vector(processed_preds_matrix.ravel()))
    _hof_corr_fingerprints.append(program_fp)
    if len(_hof_rank_pred_matrix) > _hof_max_size:
        _hof_rank_pred_matrix.pop(0)
        _hof_corr_fingerprints.pop(0)


def get_final_hof_programs() -> List[Tuple[AlphaProgram, float]]:
    """Returns the final HOF (program, mean_ic) for output/pickling."""
    # Current HOF stores (fp, fitness, ic, prog, preds_matrix)
    # The original evolve() returned List[Tuple[AlphaProgram, float]] where float was mean_ic.
    
    final_list: List[Tuple[AlphaProgram, float]] = []
    if _keep_dupes_in_hof_config:
        for _, metrics, prog in _hof_programs_data:
            final_list.append((prog, metrics.mean_ic))
            if len(final_list) >= _hof_max_size:
                break
    else:
        # _hof_programs_data is already sorted by fitness and should contain unique FPs if not _keep_dupes
        # (or best version of duplicate FPs)
        unique_output_progs: Dict[str, Tuple[AlphaProgram, float]] = {}
        for fp, metrics, prog in _hof_programs_data:
            if fp not in unique_output_progs:  # Add first encountered (best fitness)
                unique_output_progs[fp] = (prog, metrics.mean_ic)
            if len(unique_output_progs) >= _hof_max_size:
                break
        final_list = list(unique_output_progs.values())
        
    return final_list


_TOP_TO_SHOW_PRINT = 10 # From original _update_and_print_hof

def print_generation_summary(generation: int, population: List[AlphaProgram], eval_results_sorted: list): # eval_results_sorted contains (idx_in_pop, EvalResult)
    """Prints the HOF summary like _update_and_print_hof."""
    # This function will use the current state of _hof_programs_data
    
    # First, ensure _hof_programs_data is up-to-date with the current generation's best if they qualify
    # This logic was part of the original _update_and_print_hof
    # For each of the top N (e.g., _TOP_TO_SHOW_PRINT) from eval_results_sorted of current gen:
    for pop_idx, eval_res in eval_results_sorted[:_TOP_TO_SHOW_PRINT]:
        if eval_res.fitness <= -float("inf"):
            continue  # Skip invalid programs
        prog = population[pop_idx]
        fp = prog.fingerprint

    # Merge current generation's best with existing static HOF for display purposes
    # This is tricky because add_program_to_hof already updates the static HOF.
    # The printing should just reflect the current state of _hof_programs_data.

    print(f"\n★ Generation {generation+1} – Top (up to) {_TOP_TO_SHOW_PRINT} overall from HOF ★")
    hdr = " Rank | Fitness |  IC  | Ops | Finger  | First 90 chars"
    print(hdr)
    print("─" * len(hdr))
    
    # Print from the managed _hof_programs_data
    for rk, (fp, metrics, prog) in enumerate(_hof_programs_data[:_TOP_TO_SHOW_PRINT], 1):
        head = textwrap.shorten(prog.to_string(max_len=300), width=90, placeholder="…")
        print(f" {rk:>4} | {metrics.fitness:+7.4f} | {metrics.mean_ic:+5.3f} | {prog.size:3d} | {fp[:8]} | {head}")
def clear_hof():
    """Clears all HOF state."""
    global _hof_programs_data, _hof_fingerprints_set, _hof_rank_pred_matrix, _hof_corr_fingerprints
    _hof_programs_data = []
    _hof_fingerprints_set = set()
    _hof_rank_pred_matrix = []
    _hof_corr_fingerprints = []
    print("Hall of Fame cleared.")
