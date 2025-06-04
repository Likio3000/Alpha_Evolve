import dataclasses

import pytest

from config import EvolutionConfig
from evolve_alphas import evolve


def _make_cfg(n_workers: int) -> EvolutionConfig:
    return EvolutionConfig(
        data_dir="tests/data/good",
        max_lookback_data_option="common_1200",
        min_common_points=3,
        eval_lag=1,
        generations=1,
        seed=0,
        pop_size=4,
        tournament_k=2,
        elite_keep=2,
        hof_size=2,
        quiet=True,
        max_ops=5,
        n_workers=n_workers,
    )


def test_parallel_and_sequential_match():
    seq_cfg = _make_cfg(1)
    seq_result = evolve(seq_cfg)

    par_cfg = _make_cfg(2)
    par_result = evolve(par_cfg)

    assert len(seq_result) == len(par_result)
    for (prog_s, ic_s), (prog_p, ic_p) in zip(seq_result, par_result):
        assert prog_s.fingerprint == prog_p.fingerprint
        assert ic_s == pytest.approx(ic_p)

