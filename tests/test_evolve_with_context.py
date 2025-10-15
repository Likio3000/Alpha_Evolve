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
