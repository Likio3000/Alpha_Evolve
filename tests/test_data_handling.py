import pytest
import numpy as np
import pandas as pd
from collections import OrderedDict

from evolution_components.data_handling import (
    _load_and_align_data_internal,
    get_data_splits,
    initialize_data,
    get_sector_groups,
    get_features_at_time,
    configure_feature_scaling,
)
from config import DEFAULT_CRYPTO_SECTOR_MAPPING
from backtesting_components.data_handling_bt import load_and_align_data_for_backtest

DATA_DIR = "tests/data/good"
MISSING_DIR = "tests/data/missing_cols"
BAD_OVERLAP_DIR = "tests/data/bad_overlap"


def test_load_and_align_internal_success():
    aligned, common_idx, symbols = _load_and_align_data_internal(
        DATA_DIR, "common_1200", 3, 1
    )
    assert symbols == ["AAA", "BBB"]
    assert len(common_idx) == 4
    for df in aligned.values():
        assert list(df.index) == list(common_idx)
        assert "ret_1d" in df.columns
        assert "range_rel" in df.columns


def test_load_and_align_for_backtest_success():
    aligned, common_idx, symbols = load_and_align_data_for_backtest(
        DATA_DIR, "common_1200", 4
    )
    assert symbols == ["AAA", "BBB"]
    assert len(common_idx) == 4
    for df in aligned.values():
        assert list(df.index) == list(common_idx)
        assert "ret_1d" in df.columns
        assert "range_rel" in df.columns


def test_internal_missing_columns_raises():
    with pytest.raises(SystemExit):
        _load_and_align_data_internal(MISSING_DIR, "full_overlap", 3, 1)


def test_internal_insufficient_overlap_raises():
    with pytest.raises(SystemExit):
        _load_and_align_data_internal(BAD_OVERLAP_DIR, "common_1200", 3, 1)


def test_backtest_missing_columns_raises():
    with pytest.raises(SystemExit):
        load_and_align_data_for_backtest(MISSING_DIR, "full_overlap", 3)


def test_backtest_insufficient_overlap_raises():
    with pytest.raises(SystemExit):
        load_and_align_data_for_backtest(BAD_OVERLAP_DIR, "common_1200", 4)


def test_get_data_splits_returns_expected_lengths():
    initialize_data(DATA_DIR, "common_1200", 3, 1)
    train, val, test = get_data_splits(1, 1, 1)

    for split in (train, val, test):
        for df in split.values():
            assert len(df) == 2  # 1 eval step + 1 lag

    assert train["AAA"].index[-1] == val["AAA"].index[0]
    assert val["AAA"].index[-1] == test["AAA"].index[0]


def test_get_sector_groups_example_symbols():
    symbols = [
        "BINANCE_BTCUSDT, 240",
        "BINANCE_ETHUSDT, 240",
        "BYBIT_BONKUSDT, 240",
    ]
    groups = get_sector_groups(symbols)
    expected = [
        DEFAULT_CRYPTO_SECTOR_MAPPING["BTC"],
        DEFAULT_CRYPTO_SECTOR_MAPPING["ETH"],
        DEFAULT_CRYPTO_SECTOR_MAPPING["BONK"],
    ]
    assert list(groups) == expected


def test_feature_vector_scaling_zscore():
    index = pd.RangeIndex(2)
    aligned = OrderedDict({
        "A": pd.DataFrame({"opens": [1.0, 2.0]}, index=index),
        "B": pd.DataFrame({"opens": [2.0, 3.0]}, index=index),
    })
    symbols = list(aligned.keys())
    sector_vec = np.array([0, 1])
    configure_feature_scaling("zscore")
    feats = get_features_at_time(index[0], aligned, symbols, sector_vec)
    opens = feats["opens_t"]
    assert np.allclose(opens.mean(), 0.0)
    assert np.allclose(opens.std(ddof=0), 1.0)
