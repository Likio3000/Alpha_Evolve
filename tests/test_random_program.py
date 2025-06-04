import numpy as np

from alpha_framework import AlphaProgram, FINAL_PREDICTION_VECTOR_NAME


def test_random_program_final_op_vector():
    feature_vars = AlphaProgram._get_default_feature_vars()
    state_vars = {"dummy": "scalar"}
    rng = np.random.default_rng(0)
    for _ in range(20):
        prog = AlphaProgram.random_program(
            feature_vars, state_vars, max_total_ops=8, rng=rng
        )
        assert prog.predict_ops, "random_program should produce predict ops"
        last_op = prog.predict_ops[-1]
        assert last_op.out == FINAL_PREDICTION_VECTOR_NAME
        after_setup = prog._trace_vars_for_block(
            prog.setup, {**feature_vars, **state_vars}
        )
        vars_after_predict = prog._trace_vars_for_block(prog.predict_ops, after_setup)
        assert vars_after_predict[FINAL_PREDICTION_VECTOR_NAME] == "vector"
