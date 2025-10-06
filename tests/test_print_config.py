from dataclasses import asdict

from run_pipeline import parse_args


def test_pipeline_config_layering(tmp_path):
    """Config file + CLI layering should yield the expected merged config objects."""
    cfg_path = tmp_path / "pipe.toml"
    cfg_path.write_text(
        """
        [evolution]
        data_dir = "file_data"
        generations = 3
        [backtest]
        top_to_backtest = 2
        """
    )

    evo_cfg, bt_cfg, ns = parse_args(["5", "--config", str(cfg_path)])

    assert evo_cfg.generations == 5  # positional overrides file
    assert evo_cfg.data_dir == "file_data"
    assert bt_cfg.top_to_backtest == 2

    payload = {"evolution": asdict(evo_cfg), "backtest": asdict(bt_cfg)}
    assert payload["evolution"]["generations"] == 5
    assert payload["backtest"]["top_to_backtest"] == 2
