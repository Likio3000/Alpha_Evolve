import numpy as np

from alpha_framework import (
    AlphaProgram,
    Op,
    FINAL_PREDICTION_VECTOR_NAME,
    SCALAR_FEATURE_NAMES,
)
from alpha_framework.program_logic_variation import (
    mutate_program_logic,
    crossover_program_logic,
)


def build_prog_with_dead_ops() -> AlphaProgram:
    ops = [
        Op("tmp1", "vec_add_scalar", ("opens_t", "const_1")),
        Op("unused", "vec_mul_scalar", ("tmp1", "const_neg_1")),
        Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", ("tmp1",)),
    ]
    return AlphaProgram(predict_ops=ops)


def make_feature_vars():
    feature_vars = {"opens_t": "vector"}
    feature_vars.update({name: "scalar" for name in SCALAR_FEATURE_NAMES})
    return feature_vars


def test_prune_removes_unreachable_and_changes_fingerprint():
    """Ensure pruning excises dead ops and alters the program fingerprint."""
    prog = build_prog_with_dead_ops()
    fp_before = prog.fingerprint
    prog.prune()
    fp_after = prog.fingerprint

    assert len(prog.predict_ops) == 2
    assert fp_after != fp_before


def test_mutate_calls_prune_and_changes_fingerprint():
    """Confirm mutations trigger pruning so fingerprints reflect simplified logic."""
    prog = build_prog_with_dead_ops()
    fp_before = prog.fingerprint

    feature_vars = make_feature_vars()
    state_vars = {}
    rng = np.random.default_rng(0)

    mutated = mutate_program_logic(
        prog,
        feature_vars,
        state_vars,
        prob_add=0.0,
        prob_remove=0.0,
        prob_change_op=0.0,
        prob_change_inputs=0.0,
        rng=rng,
    )

    assert len(mutated.predict_ops) == 2
    assert mutated.fingerprint != fp_before


def test_crossover_calls_prune_and_changes_fingerprint():
    """Validate crossover pruning drops unused outputs in the resulting child."""
    prog_a = build_prog_with_dead_ops()
    prog_b = build_prog_with_dead_ops()

    fp_before = prog_a.fingerprint
    rng = np.random.default_rng(1)

    child = crossover_program_logic(prog_a, prog_b, rng=rng)

    outputs = [op.out for op in child.predict_ops]
    assert "unused" not in outputs
    assert child.fingerprint != fp_before
