import numpy as np
import pytest

from alpha_evolve.programs import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME
from alpha_evolve.evolution import hall_of_fame as hof
from alpha_evolve.evolution.evaluation import EvalResult


def make_prog(unique: str) -> AlphaProgram:
    op = Op(FINAL_PREDICTION_VECTOR_NAME, "sign", (unique,))
    return AlphaProgram(predict_ops=[op])


def test_high_corr_program_rejected_from_hof():
    """Reject programs whose predictions are too correlated with existing hall-of-fame entries."""
    hof.initialize_hof(
        max_size=5, keep_dupes=False, corr_penalty_weight=0.25, corr_cutoff=0.5
    )

    prog_a = make_prog("const_1")
    preds_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    hof.add_program_to_hof(
        prog_a,
        EvalResult(1.0, 0.0, 0.0, 0.0, 0.0, preds_a, 0.0, 0.0, 0.0, None),
        0,
    )
    assert len(hof._hof_programs_data) == 1

    prog_b = make_prog("const_neg_1")
    preds_b = preds_a * 2.0  # perfectly correlated with preds_a
    hof.add_program_to_hof(
        prog_b,
        EvalResult(0.9, 0.0, 0.0, 0.0, 0.0, preds_b, 0.0, 0.0, 0.0, None),
        0,
    )

    assert len(hof._hof_programs_data) == 1
    assert len(hof._hof_rank_pred_matrix) == 1

    hof.clear_hof()


def test_rank_matrix_updates_and_penalty():
    """Track correlation rank matrix updates and confirm penalty magnitude reflects weight."""
    hof.initialize_hof(
        max_size=5, keep_dupes=False, corr_penalty_weight=0.5, corr_cutoff=0.0
    )
    prog = make_prog("c")
    preds = np.array([[1.0, 2.0], [2.0, 3.0]])
    hof.update_correlation_hof(prog.fingerprint, preds)
    assert len(hof._hof_rank_pred_matrix) == 1
    penalty = hof.get_correlation_penalty_with_hof(preds.ravel())
    assert penalty == pytest.approx(0.5)
    hof.clear_hof()


def test_prediction_distance_novelty_is_finite_and_zero_for_identical():
    hof.initialize_hof(
        max_size=5, keep_dupes=False, corr_penalty_weight=0.5, corr_cutoff=0.0
    )
    preds = np.array([[1.0, 2.0], [2.0, 3.0]])
    hof.update_correlation_hof("fp_a", preds)

    same = hof.get_min_prediction_distance_with_hof(preds, probe_bars=2)
    assert same == pytest.approx(0.0)

    shifted = hof.get_min_prediction_distance_with_hof(preds + 1.0, probe_bars=2)
    assert np.isfinite(shifted)
    assert shifted > 0.0

    preds_nan = preds.copy()
    preds_nan[0, 0] = np.nan
    nan_dist = hof.get_min_prediction_distance_with_hof(preds_nan, probe_bars=2)
    assert np.isfinite(nan_dist)

    hof.clear_hof()


def test_hof_capacity_keeps_best_entries_after_sorting():
    """Keep highest-fitness programs when the Hall of Fame exceeds max size."""
    hof.initialize_hof(
        max_size=2, keep_dupes=False, corr_penalty_weight=0.25, corr_cutoff=1.1
    )
    # Keep prediction vectors sufficiently distinct to avoid correlation rejects.
    preds_a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    preds_b = np.array([[1.0, 3.0], [2.0, 5.0], [4.0, 6.0]])
    preds_c = np.array([[6.0, 1.0], [2.0, 4.0], [3.0, 5.0]])
    prog_a = make_prog("a")
    prog_b = make_prog("b")
    prog_c = make_prog("c")

    hof.add_program_to_hof(
        prog_a,
        EvalResult(0.1, 0.0, 0.0, 0.0, 0.0, preds_a, 0.0, 0.0, 0.0, None),
        0,
    )
    hof.add_program_to_hof(
        prog_b,
        EvalResult(0.2, 0.0, 0.0, 0.0, 0.0, preds_b, 0.0, 0.0, 0.0, None),
        0,
    )
    hof.add_program_to_hof(
        prog_c,
        EvalResult(0.9, 0.0, 0.0, 0.0, 0.0, preds_c, 0.0, 0.0, 0.0, None),
        0,
    )

    assert len(hof._hof_programs_data) == 2
    kept_fps = {entry.fingerprint for entry in hof._hof_programs_data}
    assert prog_c.fingerprint in kept_fps
    assert prog_b.fingerprint in kept_fps
    assert prog_a.fingerprint not in kept_fps
    hof.clear_hof()


def test_initialize_hof_resets_raw_prediction_cache():
    """Re-initializing the Hall of Fame must clear raw prediction vectors."""
    hof.initialize_hof(
        max_size=5, keep_dupes=False, corr_penalty_weight=0.25, corr_cutoff=0.5
    )
    preds = np.array([[1.0, 2.0], [2.0, 3.0]])
    hof.update_correlation_hof("fp-1", preds)
    assert len(hof._hof_raw_pred_matrix) == 1

    hof.initialize_hof(
        max_size=5, keep_dupes=False, corr_penalty_weight=0.25, corr_cutoff=0.5
    )
    assert len(hof._hof_raw_pred_matrix) == 0
    hof.clear_hof()


def test_relaxed_fill_still_rejects_near_identical_predictions():
    """Even before min_fill, near-identical predictions should be rejected."""
    hof.initialize_hof(
        max_size=5,
        keep_dupes=False,
        corr_penalty_weight=0.25,
        corr_cutoff=0.1,
        min_fill=5,
    )
    preds_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    preds_b = preds_a * 2.0
    prog_a = make_prog("fill_a")
    prog_b = make_prog("fill_b")
    hof.add_program_to_hof(
        prog_a,
        EvalResult(0.6, 0.0, 0.0, 0.0, 0.0, preds_a, 0.0, 0.0, 0.0, None),
        0,
    )
    hof.add_program_to_hof(
        prog_b,
        EvalResult(0.7, 0.0, 0.0, 0.0, 0.0, preds_b, 0.0, 0.0, 0.0, None),
        0,
    )
    assert len(hof._hof_programs_data) == 1
    hof.clear_hof()
