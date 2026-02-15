from alpha_evolve.config import EvolutionConfig
from alpha_evolve.utils.context import make_eval_context_from_dir
from alpha_evolve.evolution import engine as ea


def test_evolve_with_explicit_context(monkeypatch):
    """Evolve using a pre-built evaluation context to confirm API parity with global mode."""
    cfg = EvolutionConfig(
        data_dir="tests/data/good",
        max_lookback_data_option="common_1200",
        min_common_points=3,
        generations=1,
        pop_size=2,
        workers=1,
        quiet=True,
        seed=1,
    )

    ctx = make_eval_context_from_dir(
        data_dir=cfg.data_dir,
        strategy=cfg.max_lookback_data_option,
        min_common_points=cfg.min_common_points,
        eval_lag=cfg.eval_lag,
        dh_module=__import__("alpha_evolve.evolution.data", fromlist=["*"]),
    )

    res = ea.evolve_with_context(cfg, ctx)
    assert isinstance(res, list)


def test_evolve_with_context_is_deterministic_for_fixed_seed():
    """Running the same config twice should produce the same leading HOF fingerprints."""
    cfg = EvolutionConfig(
        data_dir="tests/data/good",
        max_lookback_data_option="common_1200",
        min_common_points=3,
        generations=2,
        pop_size=8,
        workers=1,
        quiet=True,
        seed=123,
    )
    ctx = make_eval_context_from_dir(
        data_dir=cfg.data_dir,
        strategy=cfg.max_lookback_data_option,
        min_common_points=cfg.min_common_points,
        eval_lag=cfg.eval_lag,
        dh_module=__import__("alpha_evolve.evolution.data", fromlist=["*"]),
    )
    run_a = ea.evolve_with_context(cfg, ctx)
    run_b = ea.evolve_with_context(cfg, ctx)
    fp_a = [getattr(prog, "fingerprint", "") for prog, _ in run_a[:5]]
    fp_b = [getattr(prog, "fingerprint", "") for prog, _ in run_b[:5]]
    assert fp_a == fp_b
