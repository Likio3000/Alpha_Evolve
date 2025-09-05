from __future__ import annotations

import numpy as np

from alpha_framework import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME
from alpha_framework.program_logic_variation import _finalize_predict_tail


def _feature_vars_with_scalars() -> dict[str, str]:
    fv = AlphaProgram._get_default_feature_vars()
    fv.update({"const_1": "scalar", "const_neg_1": "scalar"})
    return fv


def test_finalize_tail_renames_vector_last_op():
    rng = np.random.default_rng(0)
    fv = _feature_vars_with_scalars()
    sv: dict[str, str] = {}
    # Last op produces a vector but not named as final
    prog = AlphaProgram(
        predict_ops=[Op("tmpv", "vec_mul_scalar", ("opens_t", "const_1"))]
    )
    _finalize_predict_tail(prog, fv, sv, rng)
    assert prog.predict_ops[-1].out == FINAL_PREDICTION_VECTOR_NAME
    assert prog.predict_ops[-1].opcode == "vec_mul_scalar"


def test_finalize_tail_appends_assign_for_scalar_last_op():
    rng = np.random.default_rng(1)
    fv = _feature_vars_with_scalars()
    sv: dict[str, str] = {}
    # Last op produces a scalar; finalize should append assign_vector
    prog = AlphaProgram(predict_ops=[Op("s", "add", ("const_1", "const_1"))])
    before = len(prog.predict_ops)
    _finalize_predict_tail(prog, fv, sv, rng)
    after = len(prog.predict_ops)
    assert after == before + 1
    assert prog.predict_ops[-1].out == FINAL_PREDICTION_VECTOR_NAME
    assert prog.predict_ops[-1].opcode == "assign_vector"
    # Input should be a known vector feature
    assert prog.predict_ops[-1].inputs[0] in fv and fv[prog.predict_ops[-1].inputs[0]] == "vector"

