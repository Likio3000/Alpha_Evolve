import numpy as np
import pandas as pd
from collections import OrderedDict
import pytest

from evolution_components.evaluation_logic import (
    _safe_corr_eval,
    evaluate_program,
    initialize_evaluation_cache,
    configure_evaluation,
)
from evolution_components import hall_of_fame_manager as hof
from alpha_framework import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME


def build_simple_program(suffix: str = ""):
    ops = [
        Op(f"twos{suffix}", "add", ("const_1", "const_1")),
        Op(f"scaled{suffix}", "vec_mul_scalar", ("opens_t", f"twos{suffix}")),
        Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", (f"scaled{suffix}", "const_neg_1")),
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

    res = evaluate_program(prog, dh, hof, {})

    assert res.processed_predictions.shape == (2, 2)
    expected_score = 1.0 - 0.002 * prog.size / 32
    assert res.fitness == pytest.approx(expected_score)
    assert res.mean_ic == pytest.approx(1.0)


class CountingDH:
    def __init__(self):
        self.index = pd.RangeIndex(3)
        self.dfs = OrderedDict({
            "A": pd.DataFrame({
                "opens": [1.0, 2.0, 3.0],
                "ret_fwd": [0.1, 0.2, 0.3],
            }, index=self.index),
            "B": pd.DataFrame({
                "opens": [2.0, 3.0, 4.0],
                "ret_fwd": [0.2, 0.3, 0.4],
            }, index=self.index),
        })
        self.calls = 0

    def get_aligned_dfs(self):
        self.calls += 1
        return self.dfs

    def get_common_time_index(self):
        return self.index

    def get_stock_symbols(self):
        return list(self.dfs.keys())

    def get_n_stocks(self):
        return len(self.dfs)

    def get_eval_lag(self):
        return 1


class DummyHOF:
    def get_correlation_penalty_with_hof(self, ts):
        return 0.0


def test_eval_cache_hit():
    prog = build_simple_program("a")
    dh = CountingDH()
    hof = DummyHOF()
    initialize_evaluation_cache(max_size=2)

    first = evaluate_program(prog, dh, hof, {})
    assert dh.calls == 1

    second = evaluate_program(prog, dh, hof, {})
    assert dh.calls == 1
    assert first == second


def test_eval_cache_eviction():
    prog1 = build_simple_program("a")
    prog2 = build_simple_program("b")
    prog3 = build_simple_program("c")
    dh = CountingDH()
    hof = DummyHOF()
    initialize_evaluation_cache(max_size=2)

    evaluate_program(prog1, dh, hof, {})
    evaluate_program(prog2, dh, hof, {})
    assert dh.calls == 2

    evaluate_program(prog3, dh, hof, {})  # Evicts prog1
    assert dh.calls == 3

    evaluate_program(prog2, dh, hof, {})  # Cached
    assert dh.calls == 3

    evaluate_program(prog1, dh, hof, {})  # Needs recompute
    assert dh.calls == 4


class FlatDH:
    def __init__(self):
        self.index = pd.RangeIndex(4)
        const = {"opens": [1.0] * 4, "ret_fwd": [0.0] * 4}
        self.dfs = OrderedDict({
            "A": pd.DataFrame(const, index=self.index),
            "B": pd.DataFrame(const, index=self.index),
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


class XSCrossFlatDH:
    def __init__(self):
        self.index = pd.RangeIndex(5)
        seq = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.dfs = OrderedDict({
            "A": pd.DataFrame({"opens": seq, "ret_fwd": [0.0] * 5}, index=self.index),
            "B": pd.DataFrame({"opens": seq, "ret_fwd": [0.0] * 5}, index=self.index),
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


class TemporalFlatDH:
    def __init__(self):
        self.index = pd.RangeIndex(5)
        self.dfs = OrderedDict({
            "A": pd.DataFrame({"opens": [1.0] * 5, "ret_fwd": [0.0] * 5}, index=self.index),
            "B": pd.DataFrame({"opens": [2.0] * 5, "ret_fwd": [0.0] * 5}, index=self.index),
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


def test_early_abort_triggered():
    prog = build_simple_program("ea")
    dh = FlatDH()
    hof = DummyHOF()
    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=5e-3,
        temporal_flatness_guard=5e-3,
        early_abort_bars=3,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        scale_method="zscore",
    )
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {})
    assert res.fitness == -float("inf")
    assert res.processed_predictions is None


def test_flatness_guard_cross_sectional():
    prog = build_simple_program("xs")
    dh = XSCrossFlatDH()
    hof = DummyHOF()
    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=5e-3,
        temporal_flatness_guard=5e-3,
        early_abort_bars=100,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        scale_method="zscore",
    )
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {})
    assert res.processed_predictions.shape == (len(dh.index) - dh.get_eval_lag(), dh.get_n_stocks())
    assert res.fitness == -float("inf")


def test_flatness_guard_temporal():
    prog = build_simple_program("tmp")
    dh = TemporalFlatDH()
    hof = DummyHOF()
    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=5e-3,
        temporal_flatness_guard=5e-3,
        early_abort_bars=100,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        scale_method="zscore",
    )
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {})
    assert res.processed_predictions.shape == (len(dh.index) - dh.get_eval_lag(), dh.get_n_stocks())
    assert res.fitness == -float("inf")


def test_correlation_penalty_applied():
    prog = build_simple_program("corr")
    dh = CountingDH()
    hof.initialize_hof(max_size=5, keep_dupes=False, corr_penalty_weight=0.5, corr_cutoff=0.0)
    initialize_evaluation_cache(max_size=2)

    res1 = evaluate_program(prog, dh, hof, {})
    hof.update_correlation_hof(prog.fingerprint, res1.processed_predictions)

    initialize_evaluation_cache(max_size=2)
    res2 = evaluate_program(prog, dh, hof, {})
    base_score = 1.0 - 0.002 * prog.size / 32
    assert res2.fitness == pytest.approx(base_score - 0.5)

