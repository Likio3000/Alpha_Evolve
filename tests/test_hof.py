import numpy as np

from alpha_framework import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME
from evolution_components import hall_of_fame_manager as hof


def make_prog(unique: str) -> AlphaProgram:
    op = Op(FINAL_PREDICTION_VECTOR_NAME, "sign", (unique,))
    return AlphaProgram(predict_ops=[op])


def test_high_corr_program_rejected_from_hof():
    hof.initialize_hof(max_size=5, keep_dupes=False, corr_penalty_weight=0.25, corr_cutoff=0.5)

    prog_a = make_prog("const_1")
    preds_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    hof.add_program_to_hof(prog_a, fitness=1.0, mean_ic=0.0, processed_preds_matrix=preds_a)
    assert len(hof._hof_programs_data) == 1

    prog_b = make_prog("const_neg_1")
    preds_b = preds_a * 2.0  # perfectly correlated with preds_a
    hof.add_program_to_hof(prog_b, fitness=0.9, mean_ic=0.0, processed_preds_matrix=preds_b)

    assert len(hof._hof_programs_data) == 1
    assert len(hof._hof_processed_prediction_timeseries_for_corr) == 1

    hof.clear_hof()
