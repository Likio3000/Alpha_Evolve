#!/usr/bin/env python3
"""
Fetch 20 years of daily OHLC data for current S&P 500 constituents and
save as CSVs matching the project’s loader format: time (unix secs), open, high, low, close.

Usage:
  python scripts/fetch_sp500_data.py --out data_sp500 --years 20 [--no-auto-adjust] [--tickers path]

Dependencies:
  - pandas
  - yfinance (pip install yfinance)
  - lxml (for pandas.read_html to scrape Wikipedia)

Notes:
  - Tickers from Wikipedia have dots (e.g., BRK.B). Yahoo uses dashes (BRK-B).
    We convert automatically for fetching. Output filenames use the Yahoo form.
  - By default we auto-adjust prices (splits/dividends) for consistent OHLC.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd


def get_sp500_tickers_from_wikipedia() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    # The first table typically contains the constituents with a "Symbol" column
    for tbl in tables:
        if "Symbol" in tbl.columns:
            symbols = tbl["Symbol"].astype(str).str.strip().tolist()
            return symbols
    raise RuntimeError("Could not find 'Symbol' column while scraping Wikipedia")


def normalize_to_yahoo(tickers: List[str]) -> List[str]:
    # Yahoo uses '-' instead of '.' in tickers (e.g., BRK.B -> BRK-B)
    return [t.replace(".", "-") for t in tickers]


def load_tickers_from_file(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def ensure_deps():
    try:
        import yfinance  # noqa: F401
    except Exception as e:
        print(
            "Missing dependency: yfinance. Install with 'pip install yfinance lxml'\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def download_and_save(
    tickers: List[str], out_dir: Path, years: int, auto_adjust: bool, pause_sec: float
) -> None:
    import yfinance as yf

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(tickers, start=1):
        try:
            df = yf.download(
                t,
                period=f"{years}y",
                interval="1d",
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                print(f"[WARN] No data for {t}")
                continue

            # Keep only required OHLC columns and rename to lowercase
            if isinstance(df.columns, pd.MultiIndex):
                # yfinance now returns MultiIndex columns even for single tickers
                # with levels [Price, Ticker]. Drop the Ticker level.
                if df.columns.nlevels == 2 and set(df.columns.names) == {"Price", "Ticker"}:
                    df = df.droplevel("Ticker", axis=1)
                else:
                    # Fallback: drop the last level
                    df = df.droplevel(-1, axis=1)
            df = df[["Open", "High", "Low", "Close"]].rename(
                columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
            )

            # Reset index to a column, compute UNIX seconds, then drop original date column
            df = df.reset_index()
            # The reset index column could be named 'Date' or similar; take the first column
            date_col = df.columns[0]
            dt_utc = pd.to_datetime(df[date_col], utc=True, errors="coerce")
            df.insert(0, "time", (dt_utc.view("int64") // 10**9).astype("int64"))
            df = df.drop(columns=[date_col])

            # Drop NA rows, sort by time just in case, and ensure unique timestamps
            df = df.dropna().sort_values("time")
            df = df.loc[~df["time"].duplicated(keep="first")]

            # Write one CSV per ticker
            # Match existing naming pattern: SOURCE_TICKER, 1d.csv
            out_fp = out_dir / f"YF_{t}, 1d.csv"
            df.to_csv(out_fp, index=False)
            print(f"[{i:03d}/{len(tickers)}] Saved {out_fp}")
        except Exception as e:
            print(f"[ERROR] {t}: {e}")
        finally:
            time.sleep(pause_sec)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch S&P 500 daily OHLC data")
    p.add_argument("--out", default="data_sp500", type=Path, help="Output directory for CSVs")
    p.add_argument("--years", default=20, type=int, help="Number of years of history to fetch")
    p.add_argument("--no-auto-adjust", action="store_true", help="Disable split/dividend auto-adjustment")
    p.add_argument("--tickers", type=Path, default=None, help="Optional path to newline-separated ticker list to use (Yahoo format)")
    p.add_argument("--pause", type=float, default=0.5, help="Pause seconds between requests to be gentle")
    args = p.parse_args(argv)

    ensure_deps()

    if args.tickers is not None:
        tickers = load_tickers_from_file(args.tickers)
    else:
        # scrape Wikipedia for current constituents
        raw = get_sp500_tickers_from_wikipedia()
        tickers = normalize_to_yahoo(raw)

    if not tickers:
        print("No tickers to fetch.")
        return 1

    download_and_save(
        tickers=tickers,
        out_dir=args.out,
        years=args.years,
        auto_adjust=not args.no_auto_adjust,
        pause_sec=args.pause,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
