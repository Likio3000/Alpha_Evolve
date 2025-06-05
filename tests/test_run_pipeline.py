import sys
from run_pipeline import parse_args


def test_parse_args_defaults(monkeypatch):
    argv = ["run_pipeline.py", "5"]
    monkeypatch.setattr(sys, "argv", argv)
    evo_cfg, bt_cfg, debug, run_baselines = parse_args()
    assert evo_cfg.generations == 5
    # check a couple defaults
    assert evo_cfg.seed == 42
    assert bt_cfg.top_to_backtest == 10
    assert debug is False
    assert run_baselines is False


def test_parse_args_debug_flag(monkeypatch):
    argv = ["run_pipeline.py", "3", "--debug_prints"]
    monkeypatch.setattr(sys, "argv", argv)
    _, _, debug, run_baselines = parse_args()
    assert debug is True
    assert run_baselines is False


def test_parse_args_run_baselines(monkeypatch):
    argv = ["run_pipeline.py", "3", "--run_baselines"]
    monkeypatch.setattr(sys, "argv", argv)
    _, _, _, run_baselines = parse_args()
    assert run_baselines is True


def test_train_baselines_prints(tmp_path, capsys):
    from run_pipeline import _train_baselines

    _train_baselines("tests/data/good", tmp_path)
    captured = capsys.readouterr().out
    assert "GA tree" in captured
    assert "RankLSTM" in captured
