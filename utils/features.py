from __future__ import annotations
import numpy as np
import pandas as pd

_EPS = 1e-9

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
    for w in (5, 10, 20, 30, 60, 90):
        df[f"ma{w}"] = df["close"].rolling(w, min_periods=1).mean()
        df[f"vol{w}"] = df["close"].rolling(w, min_periods=1).std(ddof=0)
    df["range"] = df["high"] - df["low"]
    df["opens"] = df["open"]
    df["highs"] = df["high"]
    df["lows"] = df["low"]
    df["closes"] = df["close"]
    df["ranges"] = df["range"]
    df["ret_1d"] = df["close"].pct_change().fillna(0.0)
    df["ret1d"] = df["ret_1d"]
    df["range_rel"] = (df["high"] - df["low"]) / df["close"]
    df["vol_spread_5_20"] = df["vol5"] - df["vol20"]
    df["vol_spread_10_30"] = df["vol10"] - df["vol30"]
    df["vol_spread_20_60"] = df["vol20"] - df["vol60"]
    df["vol_spread_30_90"] = df["vol30"] - df["vol90"]
    df["vol_ratio_5_20"] = (
        df["vol5"].replace({np.nan: 0.0})
        / (df["vol20"].abs() + _EPS)
    ) - 1.0
    df["vol_ratio_20_60"] = (
        df["vol20"].replace({np.nan: 0.0})
        / (df["vol60"].abs() + _EPS)
    ) - 1.0
    df["trend_5_20"] = df["ma5"] - df["ma20"]
    df["intraday_ret"] = (df["close"] - df["open"]) / (df["open"] + _EPS)
    df["flow_proxy"] = (df["close"] - df["low"]) / (df["close"] + _EPS)
    df["whale_move_proxy"] = df["range_rel"].abs() * df["ret_1d"].abs()
    df["onchain_activity_proxy"] = np.abs(df["ret_1d"]) * (np.abs(df["intraday_ret"]) + np.abs(df["range_rel"]))
    df["onchain_velocity_proxy"] = df["ret_1d"].rolling(14, min_periods=1).std(ddof=0)
    df["onchain_whale_proxy"] = df["whale_move_proxy"].rolling(6, min_periods=1).mean()
    df[[
        "vol_ratio_5_20",
        "vol_ratio_20_60",
        "intraday_ret",
        "flow_proxy",
        "whale_move_proxy",
        "onchain_activity_proxy",
        "onchain_velocity_proxy",
        "onchain_whale_proxy",
    ]] = (
        df[[
            "vol_ratio_5_20",
            "vol_ratio_20_60",
            "intraday_ret",
            "flow_proxy",
            "whale_move_proxy",
            "onchain_activity_proxy",
            "onchain_velocity_proxy",
            "onchain_whale_proxy",
        ]]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    # Keep a preliminary ret_fwd to maintain the same dropna() trimming behavior
    # as earlier feature builders; alignment code will recompute it consistently.
    df["ret_fwd"] = df["close"].pct_change(periods=1).shift(-1)
    return df
