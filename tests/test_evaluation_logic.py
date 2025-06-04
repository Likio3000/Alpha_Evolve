import numpy as np
import pandas as pd
from collections import OrderedDict
import pytest

from evolution_components.evaluation_logic import _safe_corr_eval, evaluate_program
from alpha_framework import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME


def build_simple_program():
    ops = [
        Op("twos", "add", ("const_1", "const_1")),
        Op("scaled", "vec_mul_scalar", ("opens_t", "twos")),
        Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", ("scaled", "const_neg_1")),
    ]
    return AlphaProgram(setup=[], predict_ops=ops, update_ops=[])


def test_safe_corr_eval_normal():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert _safe_corr_eval(a, b) == pytest.approx(1.0)


def test_safe_corr_eval_nan_and_inf():
    a = np.array([1.0, np.nan, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    assert _safe_corr_eval(a, b) == 0.0
    a_inf = np.array([1.0, np.inf, 3.0])
    assert _safe_corr_eval(a_inf, b) == 0.0


def test_safe_corr_eval_constant():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([2.0, 2.0, 2.0])
    assert _safe_corr_eval(a, b) == 0.0


def test_evaluate_program_basic(monkeypatch):
    prog = build_simple_program()

    class DummyDH:
        def __init__(self):
            self.index = pd.RangeIndex(3)
            self.dfs = OrderedDict({
                "A": pd.DataFrame({
                    "opens": [1.0, 2.0, 3.0],
                    "highs": [0, 0, 0],
                    "lows": [0, 0, 0],
                    "closes": [0, 0, 0],
                    "ranges": [0, 0, 0],
                    "ma5": [0, 0, 0], "vol5": [0, 0, 0],
                    "ma10": [0, 0, 0], "vol10": [0, 0, 0],
                    "ma20": [0, 0, 0], "vol20": [0, 0, 0],
                    "ma30": [0, 0, 0], "vol30": [0, 0, 0],
                    "ret_fwd": [0.1, 0.2, 0.3],
                }, index=self.index),
                "B": pd.DataFrame({
                    "opens": [2.0, 3.0, 4.0],
                    "highs": [0, 0, 0],
                    "lows": [0, 0, 0],
                    "closes": [0, 0, 0],
                    "ranges": [0, 0, 0],
                    "ma5": [0, 0, 0], "vol5": [0, 0, 0],
                    "ma10": [0, 0, 0], "vol10": [0, 0, 0],
                    "ma20": [0, 0, 0], "vol20": [0, 0, 0],
                    "ma30": [0, 0, 0], "vol30": [0, 0, 0],
                    "ret_fwd": [0.2, 0.3, 0.4],
                }, index=self.index),
            })

        def get_aligned_dfs(self):
            return self.dfs

        def get_common_time_index(self):
            return self.index

        def get_stock_symbols(self):
            return list(self.dfs.keys())

        def get_n_stocks(self):
            return len(self.dfs)

        def get_eval_lag(self):
            return 1

    dh = DummyDH()

    class DummyHOF:
        def get_correlation_penalty_with_hof(self, ts):
            return 0.0

    hof = DummyHOF()

    score, mean_ic, preds = evaluate_program(prog, dh, hof, {})

    assert preds.shape == (2, 2)
    expected_score = 1.0 - 0.002 * prog.size / 32
    assert score == pytest.approx(expected_score)
    assert mean_ic == pytest.approx(1.0)
