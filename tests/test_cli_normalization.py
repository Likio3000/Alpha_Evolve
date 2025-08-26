import sys
import pytest
from backtest_evolved_alphas import parse_args


def test_backtest_cli_rejects_aliases(monkeypatch):
    argv_alias = [
        "backtest_evolved_alphas.py",
        "--data", "tests/data/good",
        "--lag", "2",
        "--data_alignment_strategy", "full_overlap",
        "--min_common_data_points", "3",
        "--top", "5",
    ]
    monkeypatch.setattr(sys, "argv", argv_alias)
    with pytest.raises(SystemExit):
        parse_args()
