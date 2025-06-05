import sys
from run_pipeline import parse_args


def test_parse_args_defaults(monkeypatch):
    argv = ["run_pipeline.py", "5"]
    monkeypatch.setattr(sys, "argv", argv)
    evo_cfg, bt_cfg, debug = parse_args()
    assert evo_cfg.generations == 5
    # check a couple defaults
    assert evo_cfg.seed == 42
    assert bt_cfg.top_to_backtest == 10
    assert debug is False


def test_parse_args_debug_flag(monkeypatch):
    argv = ["run_pipeline.py", "3", "--debug_prints"]
    monkeypatch.setattr(sys, "argv", argv)
    _, _, debug = parse_args()
    assert debug is True
