import numpy as np

from alpha_evolve.programs import (
    AlphaProgram,
    FINAL_PREDICTION_VECTOR_NAME,
)
from alpha_evolve.programs.logic_generation import (
    MAX_SETUP_OPS,
    MAX_PREDICT_OPS,
    MAX_UPDATE_OPS,
)


def test_random_program_final_op_vector():
    """Generate random programs and ensure the final predict op produces the designated vector."""
    # Default feature set plus two helpful scalar constants
    feature_vars = AlphaProgram._get_default_feature_vars()
    feature_vars.update({"const_1": "scalar", "const_neg_1": "scalar"})
    state_vars = {"dummy": "scalar"}

    rng = np.random.default_rng(0)
    for _ in range(20):
        prog = AlphaProgram.random_program(feature_vars, state_vars, max_total_ops=8, rng=rng)

        assert prog.predict_ops, "random_program should produce predict ops"

        last_op = prog.predict_ops[-1]
        assert last_op.out == FINAL_PREDICTION_VECTOR_NAME

        after_setup = prog._trace_vars_for_block(prog.setup, {**feature_vars, **state_vars})
        vars_after_predict = prog._trace_vars_for_block(prog.predict_ops, after_setup)
        assert vars_after_predict[FINAL_PREDICTION_VECTOR_NAME] == "vector"


def test_random_program_respects_block_limits():
    """Confirm random program generation obeys setup/predict/update block size limits."""
    feature_vars = AlphaProgram._get_default_feature_vars()
    feature_vars.update({"const_1": "scalar", "const_neg_1": "scalar"})
    state_vars = {"dummy": "scalar"}

    rng = np.random.default_rng(1)
    prog = AlphaProgram.random_program(
        feature_vars,
        state_vars,
        max_total_ops=200,
        rng=rng,
    )

    assert len(prog.setup) <= MAX_SETUP_OPS
    assert len(prog.predict_ops) <= MAX_PREDICT_OPS
    assert len(prog.update_ops) <= MAX_UPDATE_OPS
