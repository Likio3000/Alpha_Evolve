from config import EvolutionConfig
from alpha_framework import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME
import evolve_alphas
from evolution_components import get_final_hof_programs, clear_hof


def _fixed_program():
    return AlphaProgram(
        predict_ops=[
            Op("tmp", "vec_mul_scalar", ("ma5_t", "const_1")),
            Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", ("tmp", "const_neg_1")),
        ]
    )


def test_evolve_process(monkeypatch):
    cfg = EvolutionConfig(
        data_dir="tests/data/good",
        max_lookback_data_option="common_1200",
        min_common_points=3,
        generations=1,
        pop_size=3,
        workers=1,
        quiet=True,
    )

    monkeypatch.setattr(evolve_alphas, "_random_prog", lambda cfg: _fixed_program())
    monkeypatch.setattr(evolve_alphas, "_mutate_prog", lambda p, cfg: p)

    results = evolve_alphas.evolve(cfg)

    assert len(results) > 0
    for prog, _ in results:
        assert prog.size <= cfg.max_ops
    assert len(get_final_hof_programs()) > 0

    clear_hof()
