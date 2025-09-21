import numpy as np
import pandas as pd
from collections import OrderedDict
import pytest
from evolution_components import data_handling
from utils.features import compute_basic_features

from evolution_components.evaluation_logic import (
    _safe_corr_eval,
    evaluate_program,
    initialize_evaluation_cache,
    configure_evaluation,
    _scale_signal_for_ic,
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


def build_zero_program() -> AlphaProgram:
    ops = [
        Op("zero_scalar", "sub", ("const_1", "const_1")),
        Op(FINAL_PREDICTION_VECTOR_NAME, "vec_mul_scalar", ("opens_t", "zero_scalar")),
    ]
    return AlphaProgram(predict_ops=ops)


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

        def get_sector_groups(self, symbols=None, mapping=None, cfg=None):
            return np.arange(len(self.dfs))

        def get_features_at_time(self, timestamp, aligned_dfs, stock_symbols, sector_groups_vec):
            return data_handling.get_features_at_time(timestamp, aligned_dfs, stock_symbols, sector_groups_vec)

    dh = DummyDH()

    class DummyHOF:
        def get_correlation_penalty_with_hof(self, ts):
            return 0.0

    hof = DummyHOF()

    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=5e-3,
        temporal_flatness_guard=5e-3,
        early_abort_bars=20,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        flat_bar_threshold=0.25,
        scale_method="zscore",
    )
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {})

    assert res.processed_predictions.shape == (2, 2)
    assert res.fitness == -float("inf")


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

    def get_sector_groups(self, symbols=None, mapping=None, cfg=None):
        return np.arange(len(self.dfs))

    def get_features_at_time(self, timestamp, aligned_dfs, stock_symbols, sector_groups_vec):
        return data_handling.get_features_at_time(timestamp, aligned_dfs, stock_symbols, sector_groups_vec)



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

    def get_sector_groups(self, symbols=None, mapping=None, cfg=None):
        return np.arange(len(self.dfs))
    def get_features_at_time(self, timestamp, aligned_dfs, stock_symbols, sector_groups_vec):
        return data_handling.get_features_at_time(timestamp, aligned_dfs, stock_symbols, sector_groups_vec)


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

    def get_sector_groups(self, symbols=None, mapping=None, cfg=None):
        return np.arange(len(self.dfs))
    def get_features_at_time(self, timestamp, aligned_dfs, stock_symbols, sector_groups_vec):
        return data_handling.get_features_at_time(timestamp, aligned_dfs, stock_symbols, sector_groups_vec)


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

    def get_sector_groups(self, symbols=None, mapping=None, cfg=None):
        return np.arange(len(self.dfs))

    def get_features_at_time(self, timestamp, aligned_dfs, stock_symbols, sector_groups_vec):
        return data_handling.get_features_at_time(timestamp, aligned_dfs, stock_symbols, sector_groups_vec)

class PartialFlatBarsDH:
    def __init__(self):
        self.index = pd.RangeIndex(4)
        self.dfs = OrderedDict({
            "A": pd.DataFrame({"opens": [1.0, 1.0, 2.0, 3.0], "ret_fwd": [0.0, 0.1, 0.2, 0.3]}, index=self.index),
            "B": pd.DataFrame({"opens": [1.0, 1.0, 3.0, 4.0], "ret_fwd": [0.0, 0.1, 0.2, 0.3]}, index=self.index),
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

    def get_sector_groups(self, symbols=None, mapping=None, cfg=None):
        return np.arange(len(self.dfs))
    def get_features_at_time(self, timestamp, aligned_dfs, stock_symbols, sector_groups_vec):
        return data_handling.get_features_at_time(timestamp, aligned_dfs, stock_symbols, sector_groups_vec)


class ProcessedFlatDH:
    """Raw predictions vary but processed predictions become flat."""
    def __init__(self):
        self.index = pd.RangeIndex(4)
        self.dfs = OrderedDict({
            "A": pd.DataFrame({"opens": [1.0, 2.0, 4.0, 8.0], "ret_fwd": [0.1, 0.2, 0.3, 0.4]}, index=self.index),
            "B": pd.DataFrame({"opens": [2.0, 3.0, 5.0, 9.0], "ret_fwd": [0.2, 0.3, 0.4, 0.5]}, index=self.index),
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

    def get_sector_groups(self, symbols=None, mapping=None, cfg=None):
        return np.arange(len(self.dfs))
    def get_features_at_time(self, timestamp, aligned_dfs, stock_symbols, sector_groups_vec):
        return data_handling.get_features_at_time(timestamp, aligned_dfs, stock_symbols, sector_groups_vec)


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
        flat_bar_threshold=0.25,
        scale_method="zscore",
    )
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {})
    assert res.fitness == -float("inf")
    assert res.processed_predictions is None


def test_early_abort_flat_bar_fraction():
    prog = build_simple_program("fb")
    dh = PartialFlatBarsDH()
    hof = DummyHOF()
    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=5e-3,
        temporal_flatness_guard=5e-3,
        early_abort_bars=3,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        flat_bar_threshold=0.25,
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
        flat_bar_threshold=0.25,
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
        flat_bar_threshold=0.25,
        scale_method="zscore",
    )
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {})
    assert res.processed_predictions.shape == (len(dh.index) - dh.get_eval_lag(), dh.get_n_stocks())
    assert res.fitness == -float("inf")


def test_processed_predictions_flatness_guard():
    prog = AlphaProgram(predict_ops=[
        Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", ("opens_t",))
    ])
    dh = ProcessedFlatDH()
    hof = DummyHOF()
    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=5e-3,
        temporal_flatness_guard=5e-3,
        early_abort_bars=100,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        flat_bar_threshold=0.25,
        scale_method="sign",
    )
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {})
    assert res.processed_predictions.shape == (len(dh.index) - dh.get_eval_lag(), dh.get_n_stocks())
    assert res.fitness == -float("inf")


def test_all_zero_predictions_rejected():
    prog = build_zero_program()
    dh = CountingDH()
    hof = DummyHOF()
    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=0.0,
        temporal_flatness_guard=0.0,
        early_abort_bars=20,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        flat_bar_threshold=0.25,
        scale_method="zscore",
    )
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {})
    assert res.fitness == -float("inf")


def test_correlation_penalty_applied():
    prog = build_simple_program("corr")
    dh = CountingDH()
    hof.initialize_hof(max_size=5, keep_dupes=False, corr_penalty_weight=0.5, corr_cutoff=0.0)
    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=0.0,
        temporal_flatness_guard=0.0,
        early_abort_bars=20,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        flat_bar_threshold=0.25,
        scale_method="zscore",
    )
    initialize_evaluation_cache(max_size=2)

    res1 = evaluate_program(prog, dh, hof, {})
    hof.update_correlation_hof(prog.fingerprint, res1.processed_predictions)

    initialize_evaluation_cache(max_size=2)
    res2 = evaluate_program(prog, dh, hof, {})
    base_score = 1.0 - 0.002 * prog.size / 32
    assert res2.fitness == pytest.approx(base_score - 0.5)


def test_sector_vector_available():
    class SectorDH(CountingDH):
        def get_sector_groups(self, symbols=None, mapping=None, cfg=None):
            return np.array([0, 1])

    dh = SectorDH()
    hof = DummyHOF()
    prog = AlphaProgram(predict_ops=[
        Op(FINAL_PREDICTION_VECTOR_NAME, "vec_mul_scalar", ("sector_id_vector", "const_1"))
    ])
    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=5e-3,
        temporal_flatness_guard=5e-3,
        early_abort_bars=20,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        flat_bar_threshold=0.25,
        scale_method="zscore",
    )
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {})

    expected = _scale_signal_for_ic(np.array([0.0, 1.0]), "zscore")
    assert np.allclose(res.processed_predictions[0], expected)



def test_feature_vector_check_failure():
    prog = AlphaProgram(predict_ops=[
        Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", ("state_vec",))
    ])
    dh = CountingDH()
    hof = DummyHOF()
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {"state_vec": "vector"})
    assert res.fitness == -float("inf")
    assert res.processed_predictions is None
    assert dh.calls == 0


class SharpeDH:
    def __init__(self):
        self.index = pd.RangeIndex(4)
        self.dfs = OrderedDict({
            "A": pd.DataFrame({"opens": [1, 2, 4, 7], "ret_fwd": [0.1, 0.5, 0.1, 0.4]}, index=self.index),
            "B": pd.DataFrame({"opens": [2, 1, 3, 5], "ret_fwd": [0.2, -0.1, 0.3, 0.6]}, index=self.index),
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

    def get_sector_groups(self, symbols=None, mapping=None, cfg=None):
        return np.arange(len(self.dfs))
    def get_features_at_time(self, timestamp, aligned_dfs, stock_symbols, sector_groups_vec):
        return data_handling.get_features_at_time(timestamp, aligned_dfs, stock_symbols, sector_groups_vec)


def test_sharpe_proxy_weight_alters_fitness():
    prog = build_simple_program("sp")
    dh = SharpeDH()
    hof = DummyHOF()
    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=5e-3,
        temporal_flatness_guard=5e-3,
        early_abort_bars=20,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        flat_bar_threshold=0.25,
        scale_method="zscore",
        sharpe_proxy_weight=0.0,
    )
    initialize_evaluation_cache(max_size=2)
    res0 = evaluate_program(prog, dh, hof, {})

    configure_evaluation(
        parsimony_penalty=0.002,
        max_ops=32,
        xs_flatness_guard=5e-3,
        temporal_flatness_guard=5e-3,
        early_abort_bars=20,
        early_abort_xs=0.05,
        early_abort_t=0.05,
        flat_bar_threshold=0.25,
        scale_method="zscore",
        sharpe_proxy_weight=1.0,
    )
    initialize_evaluation_cache(max_size=2)
    res1 = evaluate_program(prog, dh, hof, {})

    assert res1.fitness == pytest.approx(res0.fitness + res1.sharpe_proxy)

def test_evaluate_program_stress_and_regime(monkeypatch):
    prog = build_simple_program("stress")

    class DummyDH:
        def __init__(self):
            raw_idx = pd.RangeIndex(20)
            raw_a = pd.DataFrame({
                "open": np.linspace(100, 120, len(raw_idx)),
                "high": np.linspace(101, 121, len(raw_idx)),
                "low": np.linspace(99, 119, len(raw_idx)),
                "close": np.linspace(100.5, 120.5, len(raw_idx)),
            }, index=raw_idx)
            raw_b = pd.DataFrame({
                "open": np.linspace(90, 110, len(raw_idx)),
                "high": np.linspace(91, 111, len(raw_idx)),
                "low": np.linspace(89, 109, len(raw_idx)),
                "close": np.linspace(90.5, 110.5, len(raw_idx)),
            }, index=raw_idx)
            feat_a = compute_basic_features(raw_a)
            feat_b = compute_basic_features(raw_b)
            self.index = raw_idx
            self.dfs = OrderedDict({
                "AAA": feat_a,
                "BBB": feat_b,
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

        def get_sector_groups(self, symbols=None, mapping=None, cfg=None):
            return np.array([0, 1])

        def get_features_at_time(self, timestamp, aligned_dfs, stock_symbols, sector_groups_vec):
            return data_handling.get_features_at_time(timestamp, aligned_dfs, stock_symbols, sector_groups_vec)

    dh = DummyDH()

    class DummyHOF:
        def get_correlation_penalty_with_hof(self, ts):
            return 0.0

    hof = DummyHOF()

    configure_evaluation(
        parsimony_penalty=0.0,
        max_ops=64,
        xs_flatness_guard=0.0,
        temporal_flatness_guard=0.0,
        early_abort_bars=0,
        early_abort_xs=0.0,
        early_abort_t=0.0,
        flat_bar_threshold=1.0,
        scale_method="rank",
        stress_penalty_weight=0.5,
        stress_fee_bps=5.0,
        stress_slippage_bps=4.0,
        stress_shock_scale=1.5,
        stress_tail_fee_bps=12.0,
        stress_tail_slippage_bps=6.0,
        stress_tail_shock_scale=2.0,
        transaction_cost_bps=15.0,
        factor_penalty_weight=0.1,
        factor_penalty_factors=("ret1d_t",),
        evaluation_horizons=(1,),
        regime_diagnostic_factors=(
            "regime_volatility_t",
            "regime_momentum_t",
            "onchain_activity_proxy_t",
        ),
    )
    initialize_evaluation_cache(max_size=2)
    res = evaluate_program(prog, dh, hof, {})

    assert isinstance(res.regime_exposures, dict)
    expected_regime = {"regime_volatility_t", "regime_momentum_t", "onchain_activity_proxy_t"}
    assert res.regime_exposures.keys() & expected_regime
    assert isinstance(res.stress_metrics, dict)
    assert any(k.startswith("base_") for k in res.stress_metrics)
    assert isinstance(res.stress_scenarios, dict)
    assert "base" in res.stress_scenarios
    assert "tail" in res.stress_scenarios
    # When turnover > 0, transaction costs should be recorded
    if res.transaction_costs:
        assert any("cost" in k for k in res.transaction_costs)
