from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Dict, Set, Any # Added Set
from dataclasses import dataclass
import textwrap # For printing HoF
import logging
from . import data

if TYPE_CHECKING:
    from alpha_evolve.programs import AlphaProgram
    from .evaluation import EvalResult

@dataclass
class HOFEntry:
    fingerprint: str
    metrics: 'EvalResult'
    program: 'AlphaProgram'
    generation: int

# Module-level state for Hall of Fame
_hof_programs_data: List[HOFEntry] = []  # Store HOF entries with generation
_hof_max_size: int = 20
_hof_fingerprints_set: Set[str] = set() # For quick check of existence
_keep_dupes_in_hof_config: bool = False # Corresponds to KEEP_DUPES_IN_HOF_CONFIG
_hof_min_fill: int = 0

# For correlation penalty
# Store rank-transformed prediction vectors for correlation checks
_hof_rank_pred_matrix: List[np.ndarray] = []
_hof_corr_fingerprints: List[str] = []  # keep order to manage eviction
_hof_raw_pred_matrix: List[np.ndarray] = []  # store raw flattened predictions for exact-match fast path
# Default correlation penalty configuration mirrors Section 9
_corr_penalty_config: Dict[str, float] = {"weight": 0.35, "cutoff": 0.15}


def initialize_hof(max_size: int, keep_dupes: bool, corr_penalty_weight: float, corr_cutoff: float, *, min_fill: int = 0):
    global _hof_programs_data, _hof_max_size, _hof_fingerprints_set, _keep_dupes_in_hof_config, _hof_min_fill
    global _hof_rank_pred_matrix, _corr_penalty_config, _hof_corr_fingerprints
    
    _hof_programs_data = []
    _hof_max_size = max_size
    _hof_fingerprints_set = set()
    _keep_dupes_in_hof_config = keep_dupes # Though original was hardcoded False
    _hof_min_fill = max(0, int(min_fill))
    
    _hof_rank_pred_matrix = []
    _hof_corr_fingerprints = []
    _hof_raw_pred_matrix = []
    _corr_penalty_config = {"weight": corr_penalty_weight, "cutoff": corr_cutoff}
    logging.getLogger(__name__).info(
        "Hall of Fame initialized: max_size=%s, keep_dupes=%s, corr_penalty_w=%s, corr_cutoff=%s, min_fill=%s",
        max_size,
        keep_dupes,
        corr_penalty_weight,
        corr_cutoff,
        _hof_min_fill,
    )

def set_correlation_penalty(weight: float | None = None, cutoff: float | None = None) -> None:
    """Dynamically update correlation-penalty configuration during evolution.

    Useful for annealing: e.g., start with zero correlation penalty for a few
    generations, then ramp up to the configured target weight.
    """
    global _corr_penalty_config
    if weight is not None:
        _corr_penalty_config["weight"] = float(weight)
    if cutoff is not None:
        _corr_penalty_config["cutoff"] = float(cutoff)

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:  # Stable, finite-safe Spearman/Pearson helper
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    a = a.astype(float, copy=False)
    b = b.astype(float, copy=False)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    if not np.isfinite(denom) or denom < 1e-9:
        return 0.0
    corr = float(np.dot(a, b) / denom)
    if not np.isfinite(corr):
        return 0.0
    return max(-1.0, min(1.0, corr))

def _rank_vector(vec: np.ndarray) -> np.ndarray:
    """Return zero-mean average-tie ranks for ``vec`` used in Spearman correlation."""
    n = vec.size
    if n == 0:
        return vec.astype(float)
    order = np.argsort(vec, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    xs = vec[order].astype(float, copy=False)
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
    ranks -= ranks.mean()
    return ranks


def get_correlation_penalty_with_hof(current_prog_flat_processed_ts: np.ndarray) -> float:
    if not _hof_rank_pred_matrix or current_prog_flat_processed_ts.std(ddof=0) < 1e-9:
        return 0.0

    cand_rank = _rank_vector(current_prog_flat_processed_ts)
    # Fast-path: if an identical or numerically equal raw vector exists, full penalty
    for raw in _hof_raw_pred_matrix:
        if raw.shape == current_prog_flat_processed_ts.shape and (
            np.array_equal(raw, current_prog_flat_processed_ts) or np.allclose(raw, current_prog_flat_processed_ts, rtol=0, atol=1e-8)
        ):
            return _corr_penalty_config["weight"]
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

def get_correlation_penalty_per_bar(processed_preds_matrix: np.ndarray) -> float:
    """Compute correlation penalty by averaging per-bar Spearman correlations.

    Expects a 2D matrix [time, cross_section]; returns a single penalty value.
    """
    if processed_preds_matrix.ndim != 2 or processed_preds_matrix.shape[0] == 0:
        return 0.0
    # Fallback to flat if no HOF entries
    if not _hof_rank_pred_matrix:
        return 0.0
    # Build candidate ranks per bar (zero-mean ranks along axis=1)
    t, n = processed_preds_matrix.shape
    # Flatten per bar and average correlation to HOF flattened per-bar shapes
    # Since HOF stores flattened whole-series ranks, approximate by comparing
    # each bar against the corresponding segments if lengths match; otherwise
    # fall back to flat behavior (rare in practice if same eval window used).
    # For a robust generic implementation, compute flat penalty as proxy.
    try:
        # Rank each bar
        corrs: list[float] = []
        for bar in range(t):
            v = processed_preds_matrix[bar, :]
            if v.size < 2 or not np.all(np.isfinite(v)):
                continue
            vr = _rank_vector(v)
            for hof_rank in _hof_rank_pred_matrix:
                # Extract trailing segment of matching length if possible
                if hof_rank.size % t == 0:
                    seg_len = hof_rank.size // t
                    if seg_len == v.size:
                        seg = hof_rank[bar * seg_len : (bar + 1) * seg_len]
                        c = abs(_safe_corr(vr, seg))
                        if not np.isnan(c) and c > _corr_penalty_config["cutoff"]:
                            corrs.append(c)
        if not corrs:
            return 0.0
        return _corr_penalty_config["weight"] * float(np.mean(corrs))
    except Exception:
        return 0.0

def get_mean_corr_component_with_hof(current_prog_flat_processed_ts: np.ndarray, cutoff: float | None = None) -> float:
    """Return the mean absolute Spearman correlation (above cutoff) with HOF entries.

    This mirrors get_correlation_penalty_with_hof but returns the unweighted
    mean correlation component so callers can apply arbitrary weights for
    logging or alternative fitness calculations.
    """
    if not _hof_rank_pred_matrix or current_prog_flat_processed_ts.std(ddof=0) < 1e-9:
        return 0.0
    cand_rank = _rank_vector(current_prog_flat_processed_ts)
    co = _corr_penalty_config["cutoff"] if cutoff is None else float(cutoff)
    corrs: List[float] = []
    for hof_rank in _hof_rank_pred_matrix:
        if len(hof_rank) != len(cand_rank):
            continue
        corr = abs(_safe_corr(cand_rank, hof_rank))
        if not np.isnan(corr) and corr > co:
            corrs.append(corr)
    if not corrs:
        return 0.0
    return float(np.mean(corrs))

def get_correlation_penalty_with_weight(current_prog_flat_processed_ts: np.ndarray, *, weight: float, cutoff: float | None = None) -> float:
    """Compute correlation penalty at a provided weight and optional cutoff.

    Useful for computing a fixed-weight fitness alongside the dynamic ramped one.
    """
    mean_corr = get_mean_corr_component_with_hof(current_prog_flat_processed_ts, cutoff=cutoff)
    if mean_corr <= 0:
        return 0.0
    return float(weight) * mean_corr

def get_rank_corr_matrix(limit: int = 10, *, absolute: bool = True):
    """Return (fingerprints, correlation-matrix) for the latest up to ``limit`` HOF rank vectors.

    The correlation is Pearson on zero-mean rank vectors (equivalent to Spearman on raw signals).
    If ``absolute`` is True, returns absolute values to reflect similarity magnitude.
    """
    n = len(_hof_rank_pred_matrix)
    if n == 0:
        return [], []
    m = min(limit, n)
    # take the most recent m entries
    ranks = _hof_rank_pred_matrix[-m:]
    fps = _hof_corr_fingerprints[-m:]
    # compute corr matrix
    out = [[0.0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if len(ranks[i]) != len(ranks[j]) or len(ranks[i]) < 2:
                c = 0.0
            else:
                c = _safe_corr(ranks[i], ranks[j])
            if absolute:
                c = abs(c)
            out[i][j] = float(c)
    return fps, out

def get_correlation_penalty_with_weight_per_bar(processed_preds_matrix: np.ndarray, *, weight: float, cutoff: float | None = None) -> float:
    """Per-bar variant mirroring get_correlation_penalty_per_bar with custom weight."""
    if processed_preds_matrix.ndim != 2 or processed_preds_matrix.shape[0] == 0:
        return 0.0
    if not _hof_rank_pred_matrix:
        return 0.0
    try:
        t, n = processed_preds_matrix.shape
        corrs: list[float] = []
        for bar in range(t):
            v = processed_preds_matrix[bar, :]
            if v.size < 2 or not np.all(np.isfinite(v)):
                continue
            vr = _rank_vector(v)
            for hof_rank in _hof_rank_pred_matrix:
                if hof_rank.size % t == 0:
                    seg_len = hof_rank.size // t
                    if seg_len == v.size:
                        seg = hof_rank[bar * seg_len : (bar + 1) * seg_len]
                        c = abs(_safe_corr(vr, seg))
                        co = _corr_penalty_config["cutoff"] if cutoff is None else float(cutoff)
                        if not np.isnan(c) and c > co:
                            corrs.append(c)
        if not corrs:
            return 0.0
        return float(weight) * float(np.mean(corrs))
    except Exception:
        return 0.0

def add_program_to_hof(
    program: AlphaProgram,
    metrics: EvalResult,
    generation: int,
):
    """Add ``program`` to the Hall of Fame and update correlation tracking.

    ``generation`` specifies which evolution cycle produced the program. The
    entry is stored along with this generation value so it can be displayed when
    printing the Hall of Fame table.

    The function keeps the main HOF ordered by fitness and maintains a
    separate list of flattened prediction time series used for correlation
    penalties.  Duplicate fingerprints are ignored in the correlation list and
    the size of both structures is capped at ``_hof_max_size``.
    """
    global _hof_programs_data, _hof_fingerprints_set, _hof_rank_pred_matrix, _hof_corr_fingerprints

    logger = logging.getLogger(__name__)

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
        enforce_cutoff = len(_hof_programs_data) >= _hof_min_fill
        for hof_rank, hof_fp in zip(_hof_rank_pred_matrix, _hof_corr_fingerprints):
            if hof_fp == fp or len(hof_rank) != len(cand_rank):
                continue
            corr = abs(_safe_corr(cand_rank, hof_rank))
            if (
                enforce_cutoff
                and not np.isnan(corr)
                and corr > _corr_penalty_config["cutoff"]
            ):
                logger.debug(
                    "HOF-reject %s vs %s | corr=%.3f > %.3f",
                    fp[:8],
                    hof_fp[:8],
                    corr,
                    _corr_penalty_config["cutoff"],
                )
                return  # Too correlated – do not add to the HOF
            elif not enforce_cutoff and not np.isnan(corr) and corr > _corr_penalty_config["cutoff"]:
                logger.debug(
                    "HOF relaxed fill %s vs %s | corr=%.3f > %.3f (current=%d < min_fill=%d)",
                    fp[:8],
                    hof_fp[:8],
                    corr,
                    _corr_penalty_config["cutoff"],
                    len(_hof_programs_data),
                    _hof_min_fill,
                )
                break
    
    # Logic for adding to _hof_programs_data (main HOF for output)
    inserted = False
    if not _keep_dupes_in_hof_config and fp in _hof_fingerprints_set:
        existing_idx = -1
        for i, entry in enumerate(_hof_programs_data):
            if entry.fingerprint == fp:
                existing_idx = i
                break
        if existing_idx != -1 and metrics.fitness > _hof_programs_data[existing_idx].metrics.fitness:
            _hof_programs_data[existing_idx] = HOFEntry(fp, metrics, program, generation)
            inserted = True
        elif existing_idx == -1:
            _hof_programs_data.append(HOFEntry(fp, metrics, program, generation))
            _hof_fingerprints_set.add(fp)
            inserted = True
    else:
        _hof_programs_data.append(HOFEntry(fp, metrics, program, generation))
        _hof_fingerprints_set.add(fp)
        inserted = True


        _hof_programs_data.sort(key=lambda x: x.metrics.fitness, reverse=True)  # Sort by fitness
    if len(_hof_programs_data) > _hof_max_size:
        removed_prog_data = _hof_programs_data.pop()
        # If we remove a unique program, ensure its fingerprint is also removed from the set
        # This needs care if multiple entries could share an fp (if _keep_dupes_in_hof_config was true)
        # For now, assuming if _keep_dupes_in_hof_config is false, fingerprints in _hof_programs_data are unique.
        if not any(item.fingerprint == removed_prog_data.fingerprint for item in _hof_programs_data):
            _hof_fingerprints_set.discard(removed_prog_data.fingerprint)


    # Logic for maintaining the list used for correlation penalty.
    if processed_preds_matrix is not None and metrics.fitness > -float("inf"):
        if fp not in _hof_corr_fingerprints:
            flat = processed_preds_matrix.ravel()
            _hof_rank_pred_matrix.append(_rank_vector(flat))
            _hof_raw_pred_matrix.append(flat.copy())
            _hof_corr_fingerprints.append(fp)
            if len(_hof_rank_pred_matrix) > _hof_max_size:
                _hof_rank_pred_matrix.pop(0)
                _hof_corr_fingerprints.pop(0)
                _hof_raw_pred_matrix.pop(0)

    if inserted:
        # Always keep HOF sorted by fitness, even on in-place updates
        _hof_programs_data.sort(key=lambda x: x.metrics.fitness, reverse=True)
        
        logger.info(
            "HOF + %-8s fit=%+.4f  IC=%+.4f  ops=%d",
            fp[:8],
            metrics.fitness,
            metrics.mean_ic,
            program.size,
        )


def update_correlation_hof(program_fp: str, processed_preds_matrix: np.ndarray):
    """Add a program's predictions to the correlation HOF, ensuring uniqueness."""
    global _hof_rank_pred_matrix, _hof_corr_fingerprints, _hof_raw_pred_matrix

    if program_fp in _hof_corr_fingerprints:
        return

    flat = processed_preds_matrix.ravel()
    _hof_rank_pred_matrix.append(_rank_vector(flat))
    _hof_raw_pred_matrix.append(flat.copy())
    _hof_corr_fingerprints.append(program_fp)
    if len(_hof_rank_pred_matrix) > _hof_max_size:
        _hof_rank_pred_matrix.pop(0)
        _hof_corr_fingerprints.pop(0)
        _hof_raw_pred_matrix.pop(0)


def get_final_hof_programs() -> List[Tuple[AlphaProgram, float]]:
    """Returns the final HOF (program, mean_ic) for output/pickling."""
    # Current HOF stores (fp, fitness, ic, prog, preds_matrix)
    # The original evolve() returned List[Tuple[AlphaProgram, float]] where float was mean_ic.
    
    final_list: List[Tuple[AlphaProgram, float]] = []
    if _keep_dupes_in_hof_config:
        for entry in _hof_programs_data:
            final_list.append((entry.program, entry.metrics.mean_ic))
            if len(final_list) >= _hof_max_size:
                break
    else:
        # _hof_programs_data is already sorted by fitness and should contain unique FPs if not _keep_dupes
        # (or best version of duplicate FPs)
        unique_output_progs: Dict[str, Tuple[AlphaProgram, float]] = {}
        for entry in _hof_programs_data:
            if entry.fingerprint not in unique_output_progs:  # Add first encountered (best fitness)
                unique_output_progs[entry.fingerprint] = (entry.program, entry.metrics.mean_ic)
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

    # Merge current generation's best with existing static HOF for display purposes
    # This is tricky because add_program_to_hof already updates the static HOF.
    # The printing should just reflect the current state of _hof_programs_data.

    logger = logging.getLogger(__name__)
    logger.info("\n★ Generation %s – Top (up to) %s overall from HOF ★", generation+1, _TOP_TO_SHOW_PRINT)
    hdr = " Rank | Fitness | Fixed |  IC  | Ops | Gen | Finger  | First 90 chars"
    logger.info(hdr)
    logger.info("─" * len(hdr))
    
    # Print from the managed _hof_programs_data
    for rk, entry in enumerate(_hof_programs_data[:_TOP_TO_SHOW_PRINT], 1):
        head = textwrap.shorten(entry.program.to_string(max_len=300), width=90, placeholder="…")
        fx = getattr(entry.metrics, "fitness_static", None)
        fx_val = fx if fx is not None else float('nan')
        logger.info(
            " %4d | %+7.4f | %+7.4f | %+5.3f | %3d | %3d | %s | %s",
            rk,
            entry.metrics.fitness,
            fx_val,
            entry.metrics.mean_ic,
            entry.program.size,
            entry.generation + 1,
            entry.fingerprint[:8],
            head,
        )
def clear_hof():
    """Clears all HOF state."""
    global _hof_programs_data, _hof_fingerprints_set, _hof_rank_pred_matrix, _hof_corr_fingerprints, _hof_raw_pred_matrix
    _hof_programs_data = []
    _hof_fingerprints_set = set()
    _hof_rank_pred_matrix = []
    _hof_corr_fingerprints = []
    _hof_raw_pred_matrix = []
    data.clear_feature_cache()
    logging.getLogger(__name__).info("Hall of Fame cleared.")


def snapshot(limit: int | None = None) -> List[Dict[str, Any]]:
    """Return a lightweight snapshot of current HOF entries for diagnostics.

    Includes fingerprint, generation (1-indexed), fitness, mean_ic, ops and a
    short program string. ``limit`` limits the number of items returned.
    """
    out: List[Dict[str, Any]] = []
    n = len(_hof_programs_data) if limit is None else min(limit, len(_hof_programs_data))
    for entry in _hof_programs_data[:n]:
        try:
            out.append({
                "fp": entry.fingerprint,
                "gen": int(entry.generation) + 1,
                "fitness": float(getattr(entry.metrics, "fitness", float("nan"))),
                "mean_ic": float(getattr(entry.metrics, "mean_ic", float("nan"))),
                "ops": int(getattr(entry.program, "size", 0)),
                "program": entry.program.to_string(max_len=180),
            })
        except Exception:
            continue
    return out

def get_hof_opcode_sets(limit: int | None = None) -> List[Set[str]]:
    """Return a list of opcode sets for programs currently in HOF.

    Useful for computing simple structural similarity/novelty proxies.
    """
    out: List[Set[str]] = []
    n = len(_hof_programs_data) if limit is None else min(limit, len(_hof_programs_data))
    for entry in _hof_programs_data[:n]:
        try:
            prog = entry.program
            ops = [*getattr(prog, 'setup', []), *getattr(prog, 'predict_ops', []), *getattr(prog, 'update_ops', [])]
            opcodes = {getattr(o, 'opcode', '') for o in ops if hasattr(o, 'opcode')}
            out.append(opcodes)
        except Exception:
            continue
    return out
