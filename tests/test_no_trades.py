from alpha_evolve.programs import (
    AlphaProgram,
    Op,
    FINAL_PREDICTION_VECTOR_NAME,
    SCALAR_FEATURE_NAMES,
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
)
from alpha_evolve.backtesting.data import load_and_align_data_for_backtest
from alpha_evolve.backtesting.core import backtest_cross_sectional_alpha


def test_constant_signal_no_trades():
    """Backtest a flat signal and confirm the engine reports the no-trades condition."""
    aligned, index, symbols = load_and_align_data_for_backtest(
        "tests/data/good", "common_1200", 4
    )
    prog = AlphaProgram(
        predict_ops=[
            Op("zero", "sub", ("const_1", "const_1")),
            Op("zeros", "vec_mul_scalar", ("opens_t", "zero")),
            Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", ("zeros", "const_1")),
        ]
    )
    metrics = backtest_cross_sectional_alpha(
        prog=prog,
        aligned_dfs=aligned,
        common_time_index=index,
        stock_symbols=symbols,
        n_stocks=len(symbols),
        fee_bps=1.0,
        lag=1,
        hold=1,
        long_short_n=0,
        scale_method="zscore",
        initial_state_vars_config={"prev_s1_vec": "vector"},
        scalar_feature_names=SCALAR_FEATURE_NAMES,
        cross_sectional_feature_vector_names=CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    )
    assert metrics.get("Error") == "No trades executed"
