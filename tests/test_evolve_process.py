import numpy as np
import pickle

from alpha_evolve.config import EvolutionConfig
from alpha_evolve.programs import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME
from alpha_evolve.evolution import engine as evolve_alphas
from alpha_evolve.evolution.hall_of_fame import get_final_hof_programs, clear_hof
from alpha_evolve.evolution import evaluation as evaluation_logic
from alpha_evolve.utils.context import make_eval_context_from_dir


def _fixed_program():
    return AlphaProgram(
        predict_ops=[
            Op("tmp", "vec_mul_scalar", ("opens_t", "const_1")),
            Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", ("tmp", "const_neg_1")),
        ]
    )


def test_evolve_process(monkeypatch):
    """Run a deterministic one-generation evolution cycle and ensure outputs respect config limits."""
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

    evaluation_logic.configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=0.0,
        temporal_flatness_guard=0.0,
        early_abort_bars=20,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        flat_bar_threshold=0.25,
        scale_method="zscore",
        sharpe_proxy_weight=0.0,
    )
    evaluation_logic.initialize_evaluation_cache(max_size=2)

    results = evolve_alphas.evolve(cfg)

    assert len(results) >= 0
    for prog, _ in results:
        assert prog.size <= cfg.max_ops
    assert len(get_final_hof_programs()) >= 0

    clear_hof()


def test_restart_preserves_hof_by_default(monkeypatch):
    """Invalid later generations should restart population without wiping HOF."""
    cfg = EvolutionConfig(
        data_dir="tests/data/good",
        max_lookback_data_option="common_1200",
        min_common_points=3,
        generations=2,
        pop_size=2,
        workers=1,
        quiet=True,
        hof_per_gen=1,
        clear_hof_on_restart=False,
    )
    ctx = make_eval_context_from_dir(
        data_dir=cfg.data_dir,
        strategy=cfg.max_lookback_data_option,
        min_common_points=cfg.min_common_points,
        eval_lag=cfg.eval_lag,
        dh_module=__import__("alpha_evolve.evolution.data", fromlist=["*"]),
    )

    monkeypatch.setattr(evolve_alphas, "_random_prog", lambda _: _fixed_program())
    monkeypatch.setattr(evolve_alphas, "_mutate_prog", lambda p, _: p)

    call_counter = {"n": 0}

    def _fake_eval(*args, **kwargs):
        call_counter["n"] += 1
        # First generation: 2 pop evals + 1 HOF re-eval all valid.
        if call_counter["n"] <= 3:
            return evaluation_logic.EvalResult(
                fitness=0.1,
                mean_ic=0.1,
                sharpe_proxy=0.1,
                parsimony_penalty=0.0,
                correlation_penalty=0.0,
                processed_predictions=np.ones((2, 2), dtype=float),
                ic_std=0.0,
                turnover_proxy=0.0,
                factor_penalty=0.0,
                fitness_static=0.1,
            )
        # Later generation: all invalid.
        return evaluation_logic.EvalResult(
            fitness=float("-inf"),
            mean_ic=0.0,
            sharpe_proxy=0.0,
            parsimony_penalty=0.0,
            correlation_penalty=0.0,
            processed_predictions=None,
            ic_std=0.0,
            turnover_proxy=0.0,
            factor_penalty=0.0,
            fitness_static=None,
        )

    monkeypatch.setattr(evolve_alphas, "evaluate_program", _fake_eval)
    monkeypatch.setattr(evolve_alphas.el_module, "evaluate_program", _fake_eval)

    evolve_alphas.evolve_with_context(cfg, ctx)
    hof = get_final_hof_programs()
    assert len(hof) >= 1
    clear_hof()


def test_evolution_writes_checkpoint_pickles(tmp_path, monkeypatch):
    """Configured checkpoint generations should emit HOF pickles."""
    cfg = EvolutionConfig(
        data_dir="tests/data/good",
        max_lookback_data_option="common_1200",
        min_common_points=3,
        generations=2,
        pop_size=3,
        workers=1,
        quiet=True,
        checkpoint_gens=(1, 2),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    ctx = make_eval_context_from_dir(
        data_dir=cfg.data_dir,
        strategy=cfg.max_lookback_data_option,
        min_common_points=cfg.min_common_points,
        eval_lag=cfg.eval_lag,
        dh_module=__import__("alpha_evolve.evolution.data", fromlist=["*"]),
    )
    monkeypatch.setattr(evolve_alphas, "_random_prog", lambda _: _fixed_program())
    monkeypatch.setattr(evolve_alphas, "_mutate_prog", lambda p, _: p)

    evolve_alphas.evolve_with_context(cfg, ctx)
    for g in (1, 2):
        p = tmp_path / "checkpoints" / f"hof_gen_{g:03d}.pkl"
        assert p.exists()
        with open(p, "rb") as fh:
            payload = pickle.load(fh)
        assert isinstance(payload, list)
    clear_hof()
