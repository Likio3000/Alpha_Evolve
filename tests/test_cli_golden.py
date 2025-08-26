import sys
from dataclasses import asdict

from backtest_evolved_alphas import parse_args as parse_bt
from run_pipeline import parse_args as parse_pipeline


def test_backtest_cli_golden_mapping(monkeypatch):
    argv = [
        "backtest_evolved_alphas.py",
        "--data_dir", "data_dir",
        "--eval_lag", "1",
        "--max_lookback_data_option", "common_1200",
        "--min_common_points", "5",
        "--top_to_backtest", "3",
        "--fee", "0.5",
        "--hold", "2",
        "--long_short_n", "2",
        "--annualization_factor", "252",
        "--scale", "rank",
        "--stop_loss_pct", "0.03",
        "--seed", "123",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cfg, _ = parse_bt()
    d = asdict(cfg)
    expected_subset = {
        "data_dir": "data_dir",
        "eval_lag": 1,
        "max_lookback_data_option": "common_1200",
        "min_common_points": 5,
        "top_to_backtest": 3,
        "fee": 0.5,
        "hold": 2,
        "long_short_n": 2,
        "annualization_factor": 252.0,
        "scale": "rank",
        "stop_loss_pct": 0.03,
        "seed": 123,
    }
    for k, v in expected_subset.items():
        assert d[k] == v


def test_pipeline_cli_golden_mapping(monkeypatch):
    argv = [
        "run_pipeline.py", "5",
        "--data_dir", "data_dir",
        "--max_lookback_data_option", "full_overlap",
        "--min_common_points", "10",
        "--eval_lag", "1",
        "--top_to_backtest", "7",
        "--fee", "1.0",
        "--hold", "3",
        "--scale", "zscore",
        "--long_short_n", "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    evo_cfg, bt_cfg, _ = parse_pipeline()

    assert evo_cfg.generations == 5
    assert bt_cfg.data_dir == "data_dir"
    assert bt_cfg.max_lookback_data_option == "full_overlap"
    assert bt_cfg.min_common_points == 10
    assert bt_cfg.eval_lag == 1
    assert bt_cfg.top_to_backtest == 7
    assert bt_cfg.fee == 1.0
    assert bt_cfg.hold == 3
    assert bt_cfg.scale == "zscore"
    assert bt_cfg.long_short_n == 0

