import logging
from config import EvolutionConfig
import evolve_alphas
from alpha_framework import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME
from evolution_components.evaluation_logic import EvalResult


def _fixed_program():
    return AlphaProgram(predict_ops=[
        Op("tmp", "vec_mul_scalar", ("opens_t", "const_1")),
        Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", ("tmp", "const_neg_1")),
    ])


def _dummy_worker(args):
    idx, _ = args
    return idx, EvalResult(0.5, 0.0, 0.0, 0.0, 0.0, None)


def test_mutation_probability_adjusts(monkeypatch, caplog):
    cfg = EvolutionConfig(
        data_dir="tests/data/good",
        max_lookback_data_option="common_1200",
        min_common_points=3,
        generations=6,
        pop_size=2,
        tournament_k=1,
        p_mut=0.1,
        p_cross=0.0,
        workers=1,
        quiet=True,
        adaptive_mutation=True,
    )

    monkeypatch.setattr(evolve_alphas, "_random_prog", lambda cfg: _fixed_program())
    monkeypatch.setattr(evolve_alphas, "_mutate_prog", lambda p, cfg: p)
    monkeypatch.setattr(evolve_alphas, "_eval_worker", _dummy_worker)

    caplog.set_level(logging.INFO)
    evolve_alphas.evolve(cfg)

    adaptive_lines = [r.message for r in caplog.records if "Adaptive probabilities" in r.message]
    assert adaptive_lines, "no adaptive log lines"
    last = adaptive_lines[-1]
    p_mut_val = float(last.split("p_mut=")[1].split()[0])
    assert p_mut_val > cfg.p_mut

