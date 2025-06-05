import numpy as np
import pandas as pd
import pytest

from baselines.ga_tree import _Node, _tree_predict, _ic, _load_all_csv


# -------------------------------------------------------------------
# _tree_predict
# -------------------------------------------------------------------
def test_tree_predict_simple(tmp_path):
    data = pd.DataFrame({
        "open": [1, 2, 3],
        "close": [4, 5, 6],
    })
    tree = _Node(np.add, _Node(None, feature="open"), _Node(None, feature="close"))
    preds = _tree_predict(tree, data)
    assert preds.tolist() == [5.0, 7.0, 9.0]


# -------------------------------------------------------------------
# _ic edge cases
# -------------------------------------------------------------------
def test_ic_zero_std_returns_zero():
    preds = np.ones(5)
    rets = np.arange(5, dtype=float)
    assert _ic(preds, rets) == 0.0


def test_ic_zero_std_ret_returns_zero():
    preds = np.arange(5, dtype=float)
    rets = np.ones(5)
    assert _ic(preds, rets) == 0.0


# -------------------------------------------------------------------
# _load_all_csv
# -------------------------------------------------------------------
def test_load_all_csv_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_all_csv(str(tmp_path))

