from __future__ import annotations

import numpy as np

from alpha_evolve.config import EvolutionConfig
from alpha_evolve.evolution import engine
from alpha_evolve.evolution.evaluation import EvalResult


def _dummy_result(score: float) -> EvalResult:
    return EvalResult(
        fitness=float(score),
        mean_ic=float(score),
        sharpe_proxy=float(score),
        parsimony_penalty=0.0,
        correlation_penalty=0.0,
        processed_predictions=np.zeros((2, 2), dtype=float),
        ic_std=0.0,
        turnover_proxy=0.0,
        factor_penalty=0.0,
        fitness_static=float(score),
    )


def test_multifidelity_parallel_path_uses_worker_pool(monkeypatch) -> None:
    cfg = EvolutionConfig(
        generations=1,
        pop_size=4,
        workers=2,
        quiet=True,
        seed=7,
        selection_metric="ramped",
        data_dir="tests/data/good",
        max_lookback_data_option="common_1200",
        min_common_points=3,
        mf_enabled=True,
        mf_initial_fraction=0.25,
        mf_promote_fraction=0.5,
        mf_min_promote=2,
    )

    init_fractions: list[float | None] = []
    pool_runs: list[dict[str, int | float | None]] = []
    current_fraction: float | None = None

    def fake_pool_init(
        data_dir: str,
        strategy: str,
        min_common_points: int,
        eval_lag: int,
        sector_mapping: dict,
        eval_state: dict | None = None,
        hof_corr_state: dict | None = None,
        eval_fraction: float | None = None,
    ) -> None:
        del data_dir, strategy, min_common_points, eval_lag, sector_mapping
        del eval_state, hof_corr_state
        nonlocal current_fraction
        current_fraction = eval_fraction
        init_fractions.append(eval_fraction)

    class FakePool:
        def __init__(self, processes: int, initializer, initargs):
            self._processes = processes
            self._initializer = initializer
            self._initargs = initargs

        def __enter__(self):
            self._initializer(*self._initargs)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, func, iterable):
            items = list(iterable)
            pool_runs.append(
                {
                    "processes": int(self._processes),
                    "work_items": int(len(items)),
                    "eval_fraction": current_fraction,
                }
            )
            return [func(item) for item in items]

    def fake_eval_worker(args):
        idx, _prog = args
        if current_fraction is None:
            score = 100.0 + float(idx)
        else:
            score = float(idx)
        return idx, _dummy_result(score)

    monkeypatch.setattr(engine, "_pool_init", fake_pool_init)
    monkeypatch.setattr(engine, "Pool", FakePool)
    monkeypatch.setattr(engine, "_eval_worker", fake_eval_worker)
    monkeypatch.setattr(engine, "slice_eval_context", lambda ctx, eval_fraction: ctx)
    monkeypatch.setattr(
        engine,
        "pbar",
        lambda iterable, **kwargs: iterable,
    )
    monkeypatch.setattr(engine, "add_program_to_hof", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        engine, "update_correlation_hof", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        engine, "print_generation_summary", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(engine, "get_final_hof_programs", lambda: [])
    monkeypatch.setattr(
        engine.el_module, "evaluate_program", lambda *args, **kwargs: _dummy_result(0.0)
    )

    engine.evolve_with_context(cfg, object())  # type: ignore[arg-type]

    assert init_fractions == [0.25, None]
    assert [run["work_items"] for run in pool_runs] == [4, 2]
    assert all(run["processes"] == 2 for run in pool_runs)
