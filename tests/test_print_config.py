import sys
import json


def test_pipeline_print_config(monkeypatch, tmp_path, capsys):
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

    argv = [
        "run_pipeline.py", "5",
        "--config", str(cfg_path),
        "--print-config",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    from run_pipeline import main
    main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert set(data.keys()) == {"evolution", "backtest"}
    assert data["evolution"]["generations"] == 5  # positional wins
    assert data["evolution"]["data_dir"] == "file_data"
    assert data["backtest"]["top_to_backtest"] == 2

