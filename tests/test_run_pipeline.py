import sys
import logging
from run_pipeline import parse_args


def test_parse_args_defaults(monkeypatch):
    argv = ["run_pipeline.py", "5"]
    monkeypatch.setattr(sys, "argv", argv)
    evo_cfg, bt_cfg, ns = parse_args()
    assert evo_cfg.generations == 5
    # check a couple defaults
    assert evo_cfg.seed == 42
    assert bt_cfg.top_to_backtest == 10
    assert bt_cfg.long_short_n == 0
    assert ns.debug_prints is False
    assert ns.run_baselines is False


def test_parse_args_debug_flag(monkeypatch):
    argv = ["run_pipeline.py", "3", "--debug_prints"]
    monkeypatch.setattr(sys, "argv", argv)
    _, _, ns = parse_args()
    assert ns.debug_prints is True
    assert ns.run_baselines is False


def test_parse_args_run_baselines(monkeypatch):
    argv = ["run_pipeline.py", "3", "--run_baselines"]
    monkeypatch.setattr(sys, "argv", argv)
    _, _, ns = parse_args()
    assert ns.run_baselines is True


def test_train_baselines_logs(tmp_path, caplog):
    from run_pipeline import _train_baselines

    caplog.set_level(logging.INFO)
    _train_baselines("tests/data/good", tmp_path)
    logged = "\n".join(r.message for r in caplog.records)
    assert "GA tree" in logged
    assert "RankLSTM" in logged
