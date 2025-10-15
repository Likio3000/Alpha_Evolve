import logging
from alpha_evolve.cli.pipeline import parse_args


def test_parse_args_defaults():
    """Verify pipeline default arguments populate evolution/backtest configs sensibly."""
    evo_cfg, bt_cfg, ns = parse_args(["5"])
    assert evo_cfg.generations == 5
    # check a couple defaults
    assert evo_cfg.seed == 42
    assert bt_cfg.top_to_backtest == 10
    assert bt_cfg.long_short_n == 0
    assert ns.debug_prints is False
    assert ns.run_baselines is False


def test_parse_args_debug_flag():
    """Ensure --debug_prints flips the appropriate namespace flags without side-effects."""
    _, _, ns = parse_args(["3", "--debug_prints"])
    assert ns.debug_prints is True
    assert ns.run_baselines is False


def test_parse_args_run_baselines():
    """Check --run_baselines enables baseline training in the parsed namespace."""
    _, _, ns = parse_args(["3", "--run_baselines"])
    assert ns.run_baselines is True


def test_train_baselines_logs(tmp_path, caplog):
    """Run baseline training helper and assert both baseline names are logged."""
    from alpha_evolve.cli.pipeline import _train_baselines

    caplog.set_level(logging.INFO)
    _train_baselines("tests/data/good", tmp_path)
    logged = "\n".join(r.message for r in caplog.records)
    assert "GA tree" in logged
    assert "RankLSTM" in logged
