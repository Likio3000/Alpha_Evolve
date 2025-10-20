#!/usr/bin/env python3
"""
Create a trimmed-down subset of the Yahoo Finance S&P 500 dataset that lives in
`data_sp500`. The subset is useful for faster local experimentation and tests.

Example:
    python scripts/make_sp500_subset.py --out data_sp500_small --tickers 25 \
        --start-date 2020-01-01 --max-rows 756
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

REQUIRED_COLS = ("time", "open", "high", "low", "close")


def _parse_date_to_epoch(date_str: str) -> int:
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def _iter_csvs(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.glob("*.csv")):
        if path.is_file():
            yield path


def make_subset(
    source_dir: Path,
    out_dir: Path,
    *,
    tickers: int,
    start_epoch: Optional[int],
    end_epoch: Optional[int],
    min_rows: int,
    max_rows: Optional[int],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0

    for csv_path in _iter_csvs(source_dir):
        if kept >= tickers:
            break

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[WARN] Could not read {csv_path.name}: {exc}", file=sys.stderr)
            continue

        if any(col not in df.columns for col in REQUIRED_COLS):
            print(f"[WARN] Skipping {csv_path.name}; missing required OHLC columns", file=sys.stderr)
            continue

        trimmed = df.copy()

        if start_epoch is not None:
            trimmed = trimmed[trimmed["time"] >= start_epoch]
        if end_epoch is not None:
            trimmed = trimmed[trimmed["time"] <= end_epoch]

        trimmed = trimmed.dropna()
        trimmed = trimmed.sort_values("time")

        if max_rows is not None and len(trimmed) > max_rows:
            trimmed = trimmed.tail(max_rows)

        if len(trimmed) < min_rows:
            continue

        trimmed.to_csv(out_dir / csv_path.name, index=False)
        kept += 1

    return kept


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a smaller S&P 500 CSV bundle from an existing one.")
    parser.add_argument("--source", type=Path, default=Path("data_sp500"), help="Directory containing full CSVs.")
    parser.add_argument("--out", type=Path, default=Path("data_sp500_small"), help="Output directory for subset CSVs.")
    parser.add_argument("--tickers", type=int, default=25, help="Number of tickers to keep.")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Earliest date to keep (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Optional latest date to keep (YYYY-MM-DD).")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=252,
        help="Minimum rows (after trimming) required to keep a ticker.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=756,
        help="Maximum rows to keep per ticker (oldest rows are dropped first).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if not args.source.exists():
        print(f"Source directory not found: {args.source}", file=sys.stderr)
        return 1

    start_epoch = _parse_date_to_epoch(args.start_date) if args.start_date else None
    end_epoch = _parse_date_to_epoch(args.end_date) if args.end_date else None

    kept = make_subset(
        args.source,
        args.out,
        tickers=args.tickers,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
    )

    if kept == 0:
        print("No tickers satisfied the criteria. Try adjusting the filters.", file=sys.stderr)
        return 2

    print(f"Saved subset with {kept} tickers at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
