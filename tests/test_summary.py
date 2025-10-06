from pathlib import Path
import json

from run_pipeline import _write_summary_json


def test_write_summary_json(tmp_path: Path):
    """Generate SUMMARY.json and verify it references key artefacts for the run."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "meta").mkdir()
    # data_alignment meta (optional)
    (run_dir / "meta" / "data_alignment.json").write_text("{}")

    # dummy pickle path
    pickle_path = run_dir / "pickles" / "evolved.pkl"
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    pickle_path.write_bytes(b"\x80\x04]")

    # dummy backtest summary CSV with a single row
    bt_dir = run_dir / "backtest_portfolio_csvs"
    bt_dir.mkdir()
    summary_csv = bt_dir / "backtest_summary_top1.csv"
    summary_csv.write_text("AlphaID,Sharpe\nAlpha_01,1.23\n")

    out = _write_summary_json(run_dir, pickle_path, summary_csv)

    assert out.exists()
    data = json.loads(out.read_text())
    assert data["run_dir"] == str(run_dir)
    assert data["programs_pickle"].endswith("evolved.pkl")
    assert data["backtest_summary_csv"].endswith("backtest_summary_top1.csv")
    assert data["backtest_summary_json"].endswith("backtest_summary_top1.json")
    assert data.get("data_alignment", "").endswith("data_alignment.json")
    assert data.get("backtested_alphas") == 1
