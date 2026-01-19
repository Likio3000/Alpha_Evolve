import sys
from dataclasses import asdict
from pathlib import Path


def test_backtest_config_layering_precedence(monkeypatch, tmp_path):
    """Validate precedence order file < env < CLI when resolving backtest configuration."""
    # Build a file config
    cfg_path = tmp_path / "bt.toml"
    cfg_path.write_text(
        """
        [backtest]
        data_dir = "file_dir"
        top_to_backtest = 1
        scale = "rank"
        fee = 0.1
        """
    )

    # Env overrides (shared and BT-specific)
    monkeypatch.setenv("AE_DATA_DIR", "env_dir")
    monkeypatch.setenv("AE_BT_TOP_TO_BACKTEST", "3")
    monkeypatch.setenv("AE_BT_SCALE", "zscore")

    # CLI overrides
    argv = [
        "backtest_evolved_alphas.py",
        "--config",
        str(cfg_path),
        "--scale",
        "sign",
        "--top_to_backtest",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    from alpha_evolve.backtesting import engine as bt_engine

    parse_bt = bt_engine.parse_args
    cfg, _ = parse_bt()
    d = asdict(cfg)

    # Precedence: file < env < CLI
    assert d["data_dir"] == "env_dir"  # env overrides file
    assert d["top_to_backtest"] == 5  # CLI overrides env
    assert d["scale"] == "sign"  # CLI overrides env and file
    assert d["fee"] == 0.1  # file retained


def test_pipeline_config_layering_precedence(monkeypatch, tmp_path):
    """Confirm pipeline config layering respects CLI overrides and shared env defaults."""
    cfg_path = tmp_path / "pipe.toml"
    cfg_path.write_text(
        """
        [evolution]
        data_dir = "file_data"
        seed = 1
        generations = 2

        [backtest]
        hold = 2
        """
    )

    # Env overrides
    monkeypatch.setenv("AE_EVO_SEED", "123")
    monkeypatch.setenv("AE_DATA_DIR", "env_data")

    argv = [
        "alpha_evolve.cli.pipeline",
        "7",
        "--config",
        str(cfg_path),
        "--data_dir",
        "cli_data",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    from alpha_evolve.cli.pipeline import parse_args as parse_pipe

    evo_cfg, bt_cfg, _ = parse_pipe()

    # Positional generations wins; CLI data_dir wins, env overrides seed
    assert evo_cfg.generations == 7
    assert Path(evo_cfg.data_dir).name == "cli_data"
    assert Path(bt_cfg.data_dir).name == "cli_data"
    assert evo_cfg.seed == 123
    # backtest hold remains from file when no override
    assert bt_cfg.hold == 2
