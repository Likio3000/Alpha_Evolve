from __future__ import annotations
import pandas as pd

def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic rolling features on a prepared OHLC dataframe.

    Expects index to be a datetime index and columns to include
    open, high, low, close. Does not compute ``ret_fwd`` here as
    alignment code recomputes it consistently after trimming.
    """
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"Missing required column(s) for compute_basic_features: {missing_list}"
        )

    df = df.copy()
    for w in (5, 10, 20, 30):
        df[f"ma{w}"] = df["close"].rolling(w, min_periods=1).mean()
        df[f"vol{w}"] = df["close"].rolling(w, min_periods=1).std(ddof=0)
    df["range"] = df["high"] - df["low"]
    df["ret_1d"] = df["close"].pct_change().fillna(0.0)
    df["range_rel"] = (df["high"] - df["low"]) / df["close"]
    # Keep a preliminary ret_fwd to maintain the same dropna() trimming behavior
    # as earlier feature builders; alignment code will recompute it consistently.
    df["ret_fwd"] = df["close"].pct_change(periods=1).shift(-1)
    return df
