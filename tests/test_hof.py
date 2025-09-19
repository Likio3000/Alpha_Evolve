import numpy as np
import pytest

from alpha_framework import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME
from evolution_components import hall_of_fame_manager as hof
from evolution_components.evaluation_logic import EvalResult


def make_prog(unique: str) -> AlphaProgram:
    op = Op(FINAL_PREDICTION_VECTOR_NAME, "sign", (unique,))
    return AlphaProgram(predict_ops=[op])


def test_high_corr_program_rejected_from_hof():
    hof.initialize_hof(max_size=5, keep_dupes=False, corr_penalty_weight=0.25, corr_cutoff=0.5)

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
    hof.initialize_hof(max_size=5, keep_dupes=False, corr_penalty_weight=0.5, corr_cutoff=0.0)
    prog = make_prog("c")
    preds = np.array([[1.0, 2.0], [2.0, 3.0]])
    hof.update_correlation_hof(prog.fingerprint, preds)
    assert len(hof._hof_rank_pred_matrix) == 1
    penalty = hof.get_correlation_penalty_with_hof(preds.ravel())
    assert penalty == pytest.approx(0.5)
    hof.clear_hof()
