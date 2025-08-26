import pickle
import sys

import pytest

from backtest_evolved_alphas import parse_args, load_programs_from_pickle
from utils.errors import BacktestError


# -------------------------------------------------------------------
# parse_args
# -------------------------------------------------------------------
def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["backtest_evolved_alphas.py"])
    cfg, ns = parse_args()
    assert cfg.top_to_backtest == 10
    assert cfg.long_short_n == 0
    assert ns.input == "evolved_top_alphas.pkl"
    assert ns.outdir == "evolved_bt_cs_results"


def test_parse_args_overrides(monkeypatch):
    argv = [
        "backtest_evolved_alphas.py",
        "--top_to_backtest", "3",
        "--fee", "0.5",
        "--scale", "rank",
        "--data_dir", "data_dir",
        "--long_short_n", "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cfg, _ = parse_args()
    assert cfg.top_to_backtest == 3
    assert cfg.fee == 0.5
    assert cfg.scale == "rank"
    assert cfg.long_short_n == 2


# -------------------------------------------------------------------
# load_programs_from_pickle
# -------------------------------------------------------------------
def test_load_programs_from_pickle_valid(tmp_path):
    data = [("prog1", 0.1), ("prog2", 0.2)]
    fp = tmp_path / "p.pkl"
    with open(fp, "wb") as f:
        pickle.dump(data, f)
    result = load_programs_from_pickle(1, str(fp))
    assert result == data[:1]


def test_load_programs_from_pickle_missing(tmp_path):
    with pytest.raises(BacktestError):
        load_programs_from_pickle(1, str(tmp_path / "missing.pkl"))


def test_load_programs_from_pickle_invalid(tmp_path):
    fp = tmp_path / "bad.pkl"
    fp.write_text("not a pickle")
    with pytest.raises(BacktestError):
        load_programs_from_pickle(1, str(fp))
