import sys
from backtest_evolved_alphas import parse_args


def test_backtest_cli_aliases_equivalent(monkeypatch):
    # Long names
    argv_long = [
        "backtest_evolved_alphas.py",
        "--data_dir", "tests/data/good",
        "--eval_lag", "2",
        "--max_lookback_data_option", "full_overlap",
        "--min_common_points", "3",
        "--top_to_backtest", "5",
    ]
    monkeypatch.setattr(sys, "argv", argv_long)
    cfg_long, _ = parse_args()

    # Aliases
    argv_alias = [
        "backtest_evolved_alphas.py",
        "--data", "tests/data/good",
        "--lag", "2",
        "--data_alignment_strategy", "full_overlap",
        "--min_common_data_points", "3",
        "--top", "5",
    ]
    monkeypatch.setattr(sys, "argv", argv_alias)
    cfg_alias, _ = parse_args()

    assert cfg_long == cfg_alias

