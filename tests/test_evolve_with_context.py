from config import EvolutionConfig
from utils.context import make_eval_context_from_dir
import evolve_alphas as ea


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
        dh_module=__import__("evolution_components.data_handling", fromlist=["*"]),
    )

    res = ea.evolve_with_context(cfg, ctx)
    assert isinstance(res, list)
