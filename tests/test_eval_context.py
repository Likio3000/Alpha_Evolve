from alpha_evolve.evolution import data as dh
from alpha_evolve.evolution.evaluation import evaluate_program
from alpha_evolve.utils.context import make_eval_context_from_globals
from alpha_evolve.programs import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME


def _simple_prog() -> AlphaProgram:
    return AlphaProgram(
        predict_ops=[
            Op("twos", "add", ("const_1", "const_1")),
            Op("scaled", "vec_mul_scalar", ("opens_t", "twos")),
            Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", ("scaled", "const_neg_1")),
        ]
    )


def test_eval_context_matches_globals():
    """Verify evaluation via explicit context mirrors evaluation through module globals."""
    # Initialize globals
    dh.initialize_data("tests/data/good", "common_1200", 3, 1)
    ctx = make_eval_context_from_globals(dh)

    prog = _simple_prog()

    # Evaluate via globals
    from alpha_evolve.evolution import hall_of_fame as hof
    metrics_global = evaluate_program(prog, dh, hof, {})

    # Evaluate via explicit context
    metrics_ctx = evaluate_program(prog, dh, hof, {}, ctx=ctx)

    assert metrics_global.fitness == metrics_ctx.fitness
    assert metrics_global.mean_ic == metrics_ctx.mean_ic
