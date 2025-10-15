from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha_evolve.programs.types import CROSS_SECTIONAL_FEATURE_VECTOR_NAMES
from alpha_evolve.evolution import evaluation as el
from alpha_evolve.evolution import hall_of_fame as hof
from alpha_evolve.evolution import data as dh


class _DummyProgram:
    fingerprint = "dummy_factor_prog"
    size = 1
    setup = []
    predict_ops = []
    update_ops = []

    def new_state(self):
        return {}

    def eval(self, features_at_t, program_state, n_stocks):
        return np.array(features_at_t["ret1d_t"], dtype=float)


@pytest.fixture(autouse=True)
def _reset_hof_and_cache():
    el.initialize_evaluation_cache(8)
    hof.initialize_hof(max_size=5, keep_dupes=False, corr_penalty_weight=0.0, corr_cutoff=0.0)
    yield
    hof.clear_hof()


def _make_context(ret1d_matrix: np.ndarray, ret_fwd_matrix: np.ndarray) -> SimpleNamespace:
    times = pd.date_range("2020-01-01", periods=ret1d_matrix.shape[0], freq="D")
    symbols = ["A", "B"]
    aligned = OrderedDict()
    for idx, sym in enumerate(symbols):
        df = pd.DataFrame({"ret_fwd": ret_fwd_matrix[:, idx]}, index=times)
        aligned[sym] = df

    col_matrix_map = {}
    for feat in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
        if feat == "sector_id_vector":
            continue
        col = feat.replace("_t", "")
        col_matrix_map[col] = np.zeros_like(ret1d_matrix)
    col_matrix_map["ret1d"] = ret1d_matrix

    bundle = SimpleNamespace(aligned_dfs=aligned, common_index=times, symbols=symbols)
    return SimpleNamespace(
        bundle=bundle,
        eval_lag=1,
        sector_ids=np.array([0, 1]),
        col_matrix_map=col_matrix_map,
    )


def _configure(factor_weight: float, factors: str) -> None:
    el.configure_evaluation(
        parsimony_penalty=0.0,
        max_ops=8,
        xs_flatness_guard=0.0,
        temporal_flatness_guard=0.0,
        early_abort_bars=10,
        early_abort_xs=0.0,
        early_abort_t=0.0,
        flat_bar_threshold=1.0,
        scale_method="zscore",
        sharpe_proxy_weight=0.0,
        ic_std_penalty_weight=0.0,
        turnover_penalty_weight=0.0,
        ic_tstat_weight=0.0,
        factor_penalty_weight=factor_weight,
        factor_penalty_factors=factors,
        use_train_val_splits=False,
        train_points=0,
        val_points=0,
        sector_neutralize=False,
        winsor_p=0.0,
        parsimony_jitter_pct=0.0,
        hof_corr_mode="flat",
        temporal_decay_half_life=0.0,
        cv_k_folds=0,
        cv_embargo=0,
    )


def test_factor_penalty_reduces_fitness(monkeypatch):
    """Verify enabling factor penalties subtracts the configured weight from fitness."""
    monkeypatch.setattr(el, "_uses_feature_vector_check", lambda prog: True)

    ret1d = np.array(
        [
            [0.10, -0.10],
            [0.20, -0.20],
            [0.15, -0.15],
            [0.05, -0.05],
        ],
        dtype=float,
    )
    ret_fwd = np.array(
        [
            [0.02, -0.01],
            [0.01, -0.02],
            [0.03, -0.03],
            [0.00, 0.00],
        ],
        dtype=float,
    )

    ctx = _make_context(ret1d, ret_fwd)
    prog = _DummyProgram()

    _configure(factor_weight=0.0, factors="ret1d_t")
    res_no_pen = el.evaluate_program(prog, dh, hof, {}, return_preds=True, ctx=ctx)

    el.initialize_evaluation_cache(8)
    _configure(factor_weight=0.5, factors="ret1d_t")
    hof.clear_hof()
    res_pen = el.evaluate_program(prog, dh, hof, {}, return_preds=True, ctx=ctx)

    assert res_no_pen.factor_penalty == pytest.approx(0.0)
    assert res_pen.factor_penalty == pytest.approx(0.5, abs=1e-6)
    assert res_pen.fitness == pytest.approx(res_no_pen.fitness - 0.5, rel=1e-6)
