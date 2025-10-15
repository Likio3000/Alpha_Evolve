"""
Lightweight helpers to index pipeline runs and read artefacts for UI use.

Functions avoid heavy deps; CSVs are parsed via the standard library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any
import csv
import json

BASE_DIR = Path("./pipeline_runs_cs").resolve()


def list_runs(base_dir: Path | str = BASE_DIR) -> List[Dict[str, Any]]:
    base = Path(base_dir)
    if not base.exists():
        return []
    runs: List[Dict[str, Any]] = []
    for p in sorted(base.glob("run_*")):
        if not p.is_dir():
            continue
        meta = {
            "run_dir": str(p),
            "summary": str(p / "SUMMARY.json"),
            "has_summary": (p / "SUMMARY.json").exists(),
        }
        runs.append(meta)
    return runs


def get_latest_run(base_dir: Path | str = BASE_DIR) -> Optional[str]:
    base = Path(base_dir)
    latest_file = base / "LATEST"
    if latest_file.exists():
        try:
            p = latest_file.read_text().strip()
            if p and Path(p).exists():
                return p
        except Exception:
            pass
    # Fallback: most recent by mtime
    candidates = [p for p in base.glob("run_*") if p.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    return str(latest)


def read_summary(run_dir: Path | str) -> Optional[Dict[str, Any]]:
    p = Path(run_dir) / "SUMMARY.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_backtest_table(run_dir: Path | str) -> List[Dict[str, Any]]:
    bt_dir = Path(run_dir) / "backtest_portfolio_csvs"
    # Prefer the largest topN summary by name sorting
    csvs = sorted(bt_dir.glob("backtest_summary_top*.csv"))
    if not csvs:
        return []
    path = csvs[-1]
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, newline="") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                rows.append(row)
    except Exception:
        return []
    return rows

