S&P 500 Daily OHLC Data (20y)

This directory is intended to hold daily OHLC data for S&P 500 constituents, formatted to match the project’s loader expectations:

- Columns: `time,open,high,low,close`
- `time`: UNIX seconds (UTC midnight for daily bars)
- One CSV per ticker (e.g., `YF_AAPL, 1d.csv`)

Populate with the helper script:

- Command: `python scripts/fetch_sp500_data.py --out data_sp500 --years 20`
- Requires: `yfinance`, `pandas`, and `lxml` (for ticker list scraping). Install with:
  `pip install yfinance lxml`

Notes

- Tickers are pulled from Wikipedia’s “List of S&P 500 companies”. Yahoo Finance uses `-` instead of `.` (e.g. `BRK-B`). The script converts automatically.
- Prices are auto-adjusted for splits/dividends so OHLC are consistent for backtesting.
- If you prefer raw prices, pass `--no-auto-adjust`.

