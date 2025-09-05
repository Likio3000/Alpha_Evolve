from __future__ import annotations

import numpy as np

from alpha_framework import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME
from alpha_framework.program_logic_variation import (
    add_op_mutation,
    remove_op_mutation,
    change_op_mutation,
    change_inputs_mutation,
)


def _base_prog() -> AlphaProgram:
    # Simple valid program with two predict ops
    ops = [
        Op("tmp", "vec_mul_scalar", ("opens_t", "const_1")),
        Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", ("tmp", "const_neg_1")),
    ]
    return AlphaProgram(predict_ops=ops)


def _feature_vars_with_scalars() -> dict[str, str]:
    fv = AlphaProgram._get_default_feature_vars()
    fv.update({"const_1": "scalar", "const_neg_1": "scalar"})
    return fv


def test_add_op_mutation_increases_setup_len():
    rng = np.random.default_rng(0)
    prog = _base_prog()
    fv = _feature_vars_with_scalars()
    sv: dict[str, str] = {}

    before = len(prog.setup)
    add_op_mutation(prog, "setup", fv, sv, rng, vector_bias=1.0, params=None)
    after = len(prog.setup)
    assert after == before + 1


def test_remove_op_mutation_decreases_predict_len_when_possible():
    rng = np.random.default_rng(1)
    prog = _base_prog()
    before = len(prog.predict_ops)
    # With 2 predict ops, one removal should reduce length by 1
    remove_op_mutation(prog, "predict", rng)
    after = len(prog.predict_ops)
    assert after == before - 1


def test_change_op_mutation_changes_opcode():
    rng = np.random.default_rng(2)
    prog = _base_prog()
    before_opcodes = [op.opcode for op in prog.predict_ops]
    change_op_mutation(prog, "predict", rng, params=None)
    after_opcodes = [op.opcode for op in prog.predict_ops]
    # At least one opcode should change
    assert before_opcodes != after_opcodes


def test_change_inputs_mutation_changes_one_input():
    rng = np.random.default_rng(3)
    # Create a setup op with scalar inputs so eligible candidates exist
    setup_ops = [Op("sout", "add", ("const_1", "const_1"))]
    prog = AlphaProgram(setup=setup_ops, predict_ops=_base_prog().predict_ops)
    fv = _feature_vars_with_scalars()
    sv: dict[str, str] = {}

    before_inputs = prog.setup[0].inputs
    change_inputs_mutation(prog, "setup", rng, fv, sv)
    after_inputs = prog.setup[0].inputs
    assert before_inputs != after_inputs

