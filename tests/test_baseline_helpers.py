import numpy as np
import pandas as pd
import pytest

from baselines.rank_lstm import _prepare_sequences, _train_linear, _predict_linear
from baselines.rsr import _prepare_graph_features


# -------------------------------------------------------------------
# rank_lstm helpers
# -------------------------------------------------------------------
def test_prepare_sequences_shapes():
    """Ensure LSTM sequence prep returns the expected window count and targets."""
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    X, y = _prepare_sequences(df, seq_len=2)
    assert X.shape == (2, 2)
    assert y.shape == (2,)


def test_train_and_predict_linear():
    """Verify closed-form linear regression learns slope/intercept and reproduces labels."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    w = _train_linear(X, y, l2=0.0)
    assert w == pytest.approx([2.0, 1.0])
    preds = _predict_linear(w, X)
    assert np.allclose(preds, y)


# -------------------------------------------------------------------
# rsr helpers
# -------------------------------------------------------------------
def test_prepare_graph_features():
    """Confirm graph feature prep yields normalized slope/gradient windows for RSR."""
    df = pd.DataFrame({"close": [1, 2, 4, 7]})
    X = _prepare_graph_features(df, seq_len=2)
    expected = np.array([[-0.5, 0.5], [-1.0, 1.0]])
    assert np.allclose(X, expected)
