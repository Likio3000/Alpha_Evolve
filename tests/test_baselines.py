import numpy as np
import pytest
from baselines.ga_tree import train_ga_tree, backtest_ga_tree
from baselines.rank_lstm import train_rank_lstm, backtest_rank_lstm
from baselines.rsr import train_rsr

DATA_DIR = "tests/data/good"


def _check_metrics(metrics):
    assert np.isfinite(metrics["IC"]) and np.isfinite(metrics["Sharpe"])


def test_ga_tree():
    metrics = train_ga_tree(DATA_DIR)
    _check_metrics(metrics)


def test_rank_lstm():
    metrics = train_rank_lstm(DATA_DIR, seq_lens=(2,), lambdas=(0.1,))
    _check_metrics(metrics)


def test_backtest_helpers():
    ga_metrics = backtest_ga_tree(DATA_DIR)
    rl_metrics = backtest_rank_lstm(DATA_DIR, seq_len=1, lmbd=0.1)
    assert ga_metrics["Sharpe"] == pytest.approx(-3.0341598716)
    assert rl_metrics["Sharpe"] == pytest.approx(2.3333333)


def test_rsr():
    metrics = train_rsr(DATA_DIR, seq_lens=(2,), lambdas=(0.1,))
    _check_metrics(metrics)
