import json
from pathlib import Path

from alpha_evolve.config import BacktestConfig
from alpha_evolve.backtesting import engine as bt
from alpha_evolve.programs import AlphaProgram, FINAL_PREDICTION_VECTOR_NAME
from alpha_evolve.programs.ops import Op


def test_end_to_end_small(tmp_path: Path):
    """Run a miniature end-to-end backtest to ensure outputs and file names align."""
    # Build a trivial program and pickle it to simulate an evolved alpha
    prog = AlphaProgram(
        predict_ops=[
            Op("twos", "add", ("const_1", "const_1")),
            Op("scaled", "vec_mul_scalar", ("opens_t", "twos")),
            Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", ("scaled", "const_neg_1")),
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pkl = run_dir / "dummy.pkl"
    import pickle
    with open(pkl, "wb") as fh:
        pickle.dump([(prog, 0.0)], fh)

    bt_cfg = BacktestConfig(
        data_dir="tests/data/good",
        max_lookback_data_option="common_1200",
        min_common_points=3,
        eval_lag=1,
        top_to_backtest=1,
        hold=1,
        long_short_n=0,
        scale="zscore",
        annualization_factor=252,
        seed=123,
    )

    outdir = run_dir / "bt"
    csv_summary = bt.run(
        bt_cfg,
        outdir=outdir,
        programs_pickle=pkl,
        debug_prints=False,
        annualization_factor_override=None,
        logger=None,
    )

    # Check outputs created
    assert csv_summary.exists()
    # And matches the expected naming for top=1
    assert (outdir / "backtest_summary_top1.csv").exists()
