#!/usr/bin/env python
"""
Backtest diagnostics plotting

Usage:
  uv run scripts/backtest_diagnostics_plot.py /path/to/run_dir

Looks for CSVs named 'alpha_XX_timeseries.csv' under
<run_dir>/backtest_portfolio_csvs and produces simple PNGs alongside.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_alpha_timeseries(csv_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]})
    ax_eq, ax_dd, ax_other = axes

    # Equity and exposure
    ax_eq.plot(df["date"], df["equity"], label="Equity", color="#1f77b4")
    if "exposure_mult" in df.columns:
        ax_eq.plot(df["date"], df["exposure_mult"], label="Exposure Mult", color="#ff7f0e", alpha=0.7)
    ax_eq.set_ylabel("Equity / Exposure")
    ax_eq.legend(loc="upper left")
    ax_eq.grid(True, alpha=0.3)

    # Drawdown
    if "drawdown" in df.columns:
        ax_dd.fill_between(df["date"], df["drawdown"], 0.0, color="#d62728", alpha=0.4)
        ax_dd.set_ylabel("Drawdown")
        ax_dd.grid(True, alpha=0.3)

    # Stops and returns
    if "stop_hits" in df.columns:
        ax_other.bar(df["date"], df["stop_hits"], label="Stop hits", color="#9467bd", width=1.0)
    if "ret_net" in df.columns:
        ax_other.plot(df["date"], df["ret_net"], label="Ret net", color="#2ca02c", alpha=0.6)
    ax_other.set_ylabel("Stops / Ret")
    ax_other.grid(True, alpha=0.3)
    ax_other.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    out_png = csv_path.with_suffix(".png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run scripts/backtest_diagnostics_plot.py /path/to/run_dir")
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    bt_dir = run_dir / "backtest_portfolio_csvs"
    if not bt_dir.exists():
        print(f"Backtest directory not found: {bt_dir}")
        sys.exit(1)
    csvs = sorted(bt_dir.glob("alpha_*_timeseries.csv"))
    if not csvs:
        print(f"No timeseries CSVs found under {bt_dir}")
        sys.exit(1)
    for c in csvs:
        out = plot_alpha_timeseries(c)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()

