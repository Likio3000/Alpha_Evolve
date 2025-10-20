Lightweight S&P 500 Subset
==========================

This directory contains a trimmed slice of the Yahoo Finance S&P 500 dump for
fast local experiments:

- 30 symbols (alphabetical selection from the full bundle)
- ~3 years of daily history (`max_rows=756`, `start_date=2020-01-01`)
- Same schema as the main dataset: `time,open,high,low,close` with UNIX seconds.

Recreate or customise it with:

```bash
uv run python scripts/make_sp500_subset.py --out data_sp500_small --tickers 30 \
  --start-date 2020-01-01 --max-rows 756 --min-rows 504
```

Tweak the parameters to trade off runtime vs overlap (e.g., fewer symbols or a
shorter window). The preset `configs/sp500_small.toml` targets this directory
and lowers `min_common_points` to `500` so alignment succeeds quickly.
