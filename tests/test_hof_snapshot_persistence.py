from pathlib import Path
import json


def test_write_hof_snapshots(monkeypatch, tmp_path: Path):
    """Confirm hall-of-fame diagnostics are persisted per generation with non-empty entries."""
    from run_pipeline import _write_hof_snapshots

    # Fake diagnostics with hof snapshots for two generations
    fake_diags = [
        {"generation": 1, "hof": [{"fp": "aaaa", "fitness": 1.0}]},
        {"generation": 2, "hof": [{"fp": "bbbb", "fitness": 2.0}]},
        {"generation": 3, "hof": []},  # empty should be skipped
    ]

    monkeypatch.setattr("utils.diagnostics.get_all", lambda: fake_diags)
    paths = _write_hof_snapshots(tmp_path)
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        data = json.loads(p.read_text())
        assert isinstance(data, list)
        assert data[0]["fp"] in {"aaaa", "bbbb"}
