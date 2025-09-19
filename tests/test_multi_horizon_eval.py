from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evolution_components import evaluation_logic as el
from evolution_components import hall_of_fame_manager as hof
from evolution_components import data_handling as dh


class _ConstantProgram:
    fingerprint = "multi_h_const"
    size = 1
    setup = []
    predict_ops = []
    update_ops = []

    def new_state(self):
        return {}

    def eval(self, features_at_t, program_state, n_stocks):
        # Long the first symbol, short the second
        return np.array([1.0, -1.0], dtype=float)


def _make_context(close_matrix: np.ndarray) -> SimpleNamespace:
    times = pd.date_range("2021-01-01", periods=close_matrix.shape[0], freq="D")
    symbols = ["AAA", "BBB"]

    aligned = OrderedDict()
    for idx, sym in enumerate(symbols):
        closes = close_matrix[:, idx]
        df = pd.DataFrame(index=times)
        df["close"] = closes
        df["open"] = closes
        df["high"] = closes
        df["low"] = closes
        df["range"] = 0.0
        df["ma5"] = closes
        df["ma10"] = closes
        df["ma20"] = closes
        df["ma30"] = closes
        df["vol5"] = 0.0
        df["vol10"] = 0.0
        df["vol20"] = 0.0
        df["vol30"] = 0.0
        returns = np.zeros_like(closes)
        returns[1:] = closes[1:] / closes[:-1] - 1.0
        df["ret1d"] = returns
        df["range_rel"] = 0.0
        df["ret_fwd"] = 0.0
        aligned[sym] = df

    bundle = SimpleNamespace(aligned_dfs=aligned, common_index=times, symbols=symbols, diagnostics=None)
    col_map = {
        "close": close_matrix,
        "ret1d": np.vstack([aligned[sym]["ret1d"].values for sym in symbols]).T,
    }
    return SimpleNamespace(
        bundle=bundle,
        eval_lag=1,
        sector_ids=np.array([0, 1]),
        col_matrix_map=col_map,
    )


def _configure(horizons):
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
        factor_penalty_weight=0.0,
        factor_penalty_factors=(),
        evaluation_horizons=horizons,
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


def test_multi_horizon_metrics(monkeypatch):
    close_matrix = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [1.21, 0.81],
            [1.331, 0.729],
            [1.4641, 0.6561],
        ],
        dtype=float,
    )
    ctx = _make_context(close_matrix)
    prog = _ConstantProgram()

    monkeypatch.setattr(el, "_uses_feature_vector_check", lambda prog: True)

    el.initialize_evaluation_cache(4)
    hof.initialize_hof(max_size=5, keep_dupes=False, corr_penalty_weight=0.0, corr_cutoff=0.0)

    _configure(horizons=(1, 2))
    result = el.evaluate_program(prog, dh, hof, {}, return_preds=True, ctx=ctx)

    assert set(result.horizon_metrics.keys()) == {1, 2}
    # Each horizon should have perfect ranking correlation with the constant long/short predictions
    assert result.horizon_metrics[1]["mean_ic"] == pytest.approx(1.0, rel=1e-6)
    assert result.horizon_metrics[2]["mean_ic"] == pytest.approx(1.0, rel=1e-6)
    assert result.mean_ic == pytest.approx(1.0, rel=1e-6)
    # Sharpe proxy is averaged across horizons; both horizons deliver identical pnl patterns
    assert result.sharpe_proxy == pytest.approx(result.horizon_metrics[1]["sharpe"], rel=1e-6)

    hof.clear_hof()
