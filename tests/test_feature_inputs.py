import numpy as np
import pandas as pd
from collections import OrderedDict

from utils.features import compute_basic_features
from evolution_components import data_handling


def _make_price_series(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1, size=n)) + 100.0
    high = base + rng.uniform(0.1, 1.0, size=n)
    low = base - rng.uniform(0.1, 1.0, size=n)
    open_ = base + rng.normal(0, 0.2, size=n)
    close = base + rng.normal(0, 0.2, size=n)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }, index=pd.RangeIndex(n))


def test_compute_basic_features_includes_extended_columns():
    """Ensure feature engineering emits all extended technical columns with finite values."""
    raw = _make_price_series()
    out = compute_basic_features(raw)
    expected_cols = {
        "ma60", "ma90",
        "vol60", "vol90",
        "vol_spread_20_60", "vol_spread_30_90",
        "vol_ratio_20_60",
        "onchain_activity_proxy", "onchain_velocity_proxy", "onchain_whale_proxy",
    }
    missing = expected_cols.difference(out.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert np.isfinite(out[list(expected_cols)].to_numpy()).all()


def test_get_features_at_time_emits_new_vectors():
    """Confirm runtime feature extraction provides sector-aware vectors for new derived metrics."""
    raw_a = _make_price_series(seed=1)
    raw_b = _make_price_series(seed=2)
    feat_a = compute_basic_features(raw_a)
    feat_b = compute_basic_features(raw_b)
    aligned = OrderedDict({
        "AAA": feat_a,
        "BBB": feat_b,
    })
    ts = feat_a.index[50]
    sector_groups = np.array([0, 1])
    feature_dict = data_handling.get_features_at_time(ts, aligned, ["AAA", "BBB"], sector_groups)
    for key in (
        "vol_spread_20_60_t",
        "vol_spread_30_90_t",
        "vol_ratio_20_60_t",
        "onchain_activity_proxy_t",
        "onchain_velocity_proxy_t",
        "onchain_whale_proxy_t",
        "market_dispersion_t",
        "cross_btc_momentum_t",
        "sector_momentum_diff_t",
    ):
        assert key in feature_dict, f"Feature {key} missing"
        vec = feature_dict[key]
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (2,)
        assert np.isfinite(vec).all()
