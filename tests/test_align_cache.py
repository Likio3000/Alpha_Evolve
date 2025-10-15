import os
from alpha_evolve.utils.cache import (
    compute_align_cache_key,
    load_aligned_bundle_from_cache,
    save_aligned_bundle_to_cache,
)
from alpha_evolve.utils.data_loading import DataBundle, DataDiagnostics
from collections import OrderedDict
import pandas as pd


def test_compute_key_and_roundtrip(tmp_path):
    """Ensure alignment cache keys are stable and serialized bundles reload unchanged."""
    # Use tests/data/good directory for inputs
    data_dir = "tests/data/good"
    key = compute_align_cache_key(
        data_dir=data_dir,
        feature_fn_name="feat",
        strategy="common_1200",
        min_common_points=3,
        eval_lag=1,
        include_lag_in_required_length=True,
        fixed_trim_include_lag=True,
    )
    assert isinstance(key, str) and len(key) >= 10

    # Force cache dir to temp for isolation
    os.environ["AE_ALIGN_CACHE_DIR"] = str(tmp_path)
    # Build a tiny bundle to persist
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2]}, index=idx)
    bundle = DataBundle(
        aligned_dfs=OrderedDict({"AAA": df}),
        common_index=idx,
        symbols=["AAA"],
        diagnostics=DataDiagnostics(1, 1, [], 2, idx.min(), idx.max()),
    )
    save_aligned_bundle_to_cache(key, bundle)
    loaded = load_aligned_bundle_from_cache(key)
    assert loaded is not None
    assert list(loaded.symbols) == ["AAA"]
