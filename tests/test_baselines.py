import numpy as np
from baselines.ga_tree import train_ga_tree
from baselines.rank_lstm import train_rank_lstm
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


def test_rsr():
    metrics = train_rsr(DATA_DIR, seq_lens=(2,), lambdas=(0.1,))
    _check_metrics(metrics)
