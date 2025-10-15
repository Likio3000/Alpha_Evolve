import numpy as np

from alpha_evolve.programs import (
    AlphaProgram,
    Op,
    FINAL_PREDICTION_VECTOR_NAME,
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    SCALAR_FEATURE_NAMES,
)
from alpha_evolve.programs.logic_variation import (
    mutate_program_logic,
    crossover_program_logic,
)


def build_prog_a() -> AlphaProgram:
    ops = [
        Op("a1", "vec_add_scalar", ("opens_t", "const_1")),
        Op(FINAL_PREDICTION_VECTOR_NAME, "vec_mul_scalar", ("a1", "const_neg_1")),
    ]
    return AlphaProgram(predict_ops=ops)


def build_prog_b() -> AlphaProgram:
    ops = [
        Op("b1", "vec_mul_scalar", ("opens_t", "const_1")),
        Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", ("b1", "const_neg_1")),
    ]
    return AlphaProgram(predict_ops=ops)


def make_feature_dict(n: int):
    feats = {name: np.arange(1, n + 1, dtype=float) for name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES}
    for name in SCALAR_FEATURE_NAMES:
        feats[name] = 1.0 if "neg" not in name else -1.0
    return feats


def make_feature_vars():
    feature_vars = {name: "vector" for name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES}
    feature_vars.update({name: "scalar" for name in SCALAR_FEATURE_NAMES})
    return feature_vars


def test_mutate_and_crossover_eval():
    """Exercise mutation and crossover flows to ensure resulting programs evaluate correctly."""
    n = 4
    features = make_feature_dict(n)
    feature_vars = make_feature_vars()
    state_vars = {}

    base_a = build_prog_a()
    base_b = build_prog_b()

    rng = np.random.default_rng(0)
    mutated = mutate_program_logic(base_a, feature_vars, state_vars, rng=rng)
    pred = mutated.eval(features, {}, n)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (n,)
    vars_after = mutated.get_vars_at_point("predict", len(mutated.predict_ops), feature_vars, state_vars)
    assert vars_after[FINAL_PREDICTION_VECTOR_NAME] == "vector"

    rng = np.random.default_rng(1)
    crossed = crossover_program_logic(base_a, base_b, rng=rng)
    pred_c = crossed.eval(features, {}, n)
    assert isinstance(pred_c, np.ndarray)
    assert pred_c.shape == (n,)
    vars_after_c = crossed.get_vars_at_point("predict", len(crossed.predict_ops), feature_vars, state_vars)
    assert vars_after_c[FINAL_PREDICTION_VECTOR_NAME] == "vector"
