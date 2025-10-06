import pickle
import sys

import pytest

from backtest_evolved_alphas import parse_args, load_programs_from_pickle
from utils.errors import BacktestError


# -------------------------------------------------------------------
# parse_args
# -------------------------------------------------------------------
def test_parse_args_defaults(monkeypatch):
    """Assert CLI defaults supply sensible filenames and backtest parameters when no args given."""
    monkeypatch.setattr(sys, "argv", ["backtest_evolved_alphas.py"])
    cfg, ns = parse_args()
    assert cfg.top_to_backtest == 10
    assert cfg.long_short_n == 0
    assert ns.input == "evolved_top_alphas.pkl"
    assert ns.outdir == "evolved_bt_cs_results"


def test_parse_args_overrides(monkeypatch):
    """Check explicit CLI overrides correctly propagate into the parsed backtest config."""
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
    """Validate that the loader slices a pickle of programs down to the requested count."""
    data = [("prog1", 0.1), ("prog2", 0.2)]
    fp = tmp_path / "p.pkl"
    with open(fp, "wb") as f:
        pickle.dump(data, f)
    result = load_programs_from_pickle(1, str(fp))
    assert result == data[:1]


def test_load_programs_from_pickle_missing(tmp_path):
    """Ensure missing pickle files raise BacktestError for clear operator feedback."""
    with pytest.raises(BacktestError):
        load_programs_from_pickle(1, str(tmp_path / "missing.pkl"))


def test_load_programs_from_pickle_invalid(tmp_path):
    """Confirm malformed pickle contents surface BacktestError during load."""
    fp = tmp_path / "bad.pkl"
    fp.write_text("not a pickle")
    with pytest.raises(BacktestError):
        load_programs_from_pickle(1, str(fp))
