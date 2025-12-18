import json
from pathlib import Path


from alpha_evolve.cli.pipeline import _write_data_alignment_meta
from alpha_evolve.utils.data_loading import DataDiagnostics
from alpha_evolve.config import EvolutionConfig


def test_write_data_alignment_meta(monkeypatch, tmp_path: Path):
    """Persist alignment metadata alongside a run and verify serialized fields match config."""
    run_dir = tmp_path / "run"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)

    evo_cfg = EvolutionConfig(
        data_dir="tests/data/good",
        max_lookback_data_option="common_1200",
        min_common_points=3,
        eval_lag=1,
        generations=1,
    )

    di = DataDiagnostics(
        n_symbols_before=3,
        n_symbols_after=2,
        dropped_symbols=["CCC"],
        overlap_len=4,
        overlap_start=None,
        overlap_end=None,
    )

    # Monkeypatch diagnostics getter
    class DummyDH:
        @staticmethod
        def get_data_diagnostics():
            return di

    monkeypatch.setitem(globals(), "__dummy", None)  # ensure globals available
    monkeypatch.setattr("alpha_evolve.evolution.data.get_data_diagnostics", lambda: di)

    out = _write_data_alignment_meta(run_dir, evo_cfg)
    assert out.exists()

    payload = json.loads(out.read_text())
    assert payload["data_dir"] == evo_cfg.data_dir
    assert payload["strategy"] == evo_cfg.max_lookback_data_option
    assert payload["min_common_points"] == evo_cfg.min_common_points
    assert payload["eval_lag"] == evo_cfg.eval_lag
    assert payload["n_symbols_before"] == 3
    assert payload["n_symbols_after"] == 2
    assert payload["dropped_symbols"] == ["CCC"]
    assert payload["overlap_len"] == 4
