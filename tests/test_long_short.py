import numpy as np
import pandas as pd
from collections import OrderedDict
import pytest

from backtesting_components.core_logic import backtest_cross_sectional_alpha, _scale_signal_cross_sectionally

class DummyProg:
    def __init__(self, signals):
        self.signals = signals
        self.step = 0
    def new_state(self):
        self.step = 0
        return {}
    def eval(self, features_at_t, state, n_stocks):
        res = self.signals[self.step]
        self.step += 1
        return res
    @property
    def size(self):
        return 0
    def to_string(self, max_len=1000):
        return "dummy"


def _build_df(rets, index):
    cols = [
        "open",
        "high",
        "low",
        "close",
        "ma5",
        "vol5",
        "ma10",
        "vol10",
        "ma20",
        "vol20",
        "ma30",
        "vol30",
        "range",
        "ret_1d",
        "range_rel",
        "ret_fwd",
    ]
    df = pd.DataFrame(0.0, index=index, columns=cols)
    df["ret_fwd"] = rets
    return df


def manual_backtest(signals, rets, long_short_n):
    n_steps, n_stocks = signals.shape
    pos = np.zeros_like(signals)
    for t in range(n_steps):
        scaled = _scale_signal_cross_sectionally(signals[t], "zscore")
        if long_short_n > 0:
            k = min(long_short_n, n_stocks // 2)
            order = np.argsort(scaled)
            ls = np.zeros_like(scaled)
            ls[order[-k:]] = 1.0
            ls[order[:k]] = -1.0
            scaled = ls
        centered = scaled - np.mean(scaled)
        sa = np.sum(np.abs(centered))
        pos[t] = centered / sa if sa > 1e-9 else 0
    returns = np.sum(pos * rets, axis=1)
    return pos, returns


def test_long_short_n_trades_only_requested_symbols():
    index = pd.date_range("2020-01-01", periods=5)
    rets = np.array([
        [0.01, -0.02, 0.03],
        [-0.01, 0.02, 0.01],
        [0.03, 0.01, -0.02],
        [0.02, -0.01, 0.02],
        [0.0, 0.0, 0.0],
    ])
    aligned = OrderedDict({
        "AAA": _build_df(rets[:, 0], index),
        "BBB": _build_df(rets[:, 1], index),
        "CCC": _build_df(rets[:, 2], index),
    })
    signals = np.array([
        [0.1, 0.2, -0.3],
        [0.4, -0.1, 0.2],
        [0.3, 0.5, 0.1],
        [-0.2, 0.2, 0.0],
    ])
    prog = DummyProg(signals)
    metrics = backtest_cross_sectional_alpha(
        prog=prog,
        aligned_dfs=aligned,
        common_time_index=index,
        stock_symbols=list(aligned.keys()),
        n_stocks=3,
        fee_bps=0.0,
        lag=0,
        hold=1,
        scale_method="zscore",
        long_short_n=1,
        initial_state_vars_config={},
        scalar_feature_names=[],
        cross_sectional_feature_vector_names=[],
    )
    manual_pos, manual_ret = manual_backtest(signals, rets[:-1], 1)
    assert all(np.count_nonzero(p) == 2 for p in manual_pos)
    mean_ret = manual_ret.mean()
    std_ret = manual_ret.std(ddof=0)
    sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(365 * 6)
    assert metrics["Sharpe"] == pytest.approx(sharpe)

