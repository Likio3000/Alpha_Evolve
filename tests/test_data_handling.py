import pandas as pd
import pytest

from evolution_components.data_handling import (
    _load_and_align_data_internal,
    get_data_splits,
    initialize_data,
    get_sector_groups,
)
from config import DEFAULT_CRYPTO_SECTOR_MAPPING
from backtesting_components.data_handling_bt import load_and_align_data_for_backtest
from utils.data_loading_common import DataLoadError
from utils.features import compute_basic_features

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
    with pytest.raises(DataLoadError):
        _load_and_align_data_internal(MISSING_DIR, "full_overlap", 3, 1)


def test_internal_insufficient_overlap_raises():
    with pytest.raises(DataLoadError):
        _load_and_align_data_internal(BAD_OVERLAP_DIR, "common_1200", 3, 1)


def test_backtest_missing_columns_raises():
    with pytest.raises(DataLoadError):
        load_and_align_data_for_backtest(MISSING_DIR, "full_overlap", 3)


def test_backtest_insufficient_overlap_raises():
    with pytest.raises(DataLoadError):
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


@pytest.mark.parametrize(
    "missing_cols",
    [[], ["open"], ["high"], ["low"], ["close"]],
)
def test_compute_basic_features(missing_cols):
    base_df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
        }
    )
    df = base_df.drop(columns=missing_cols)
    if missing_cols:
        with pytest.raises(ValueError):
            compute_basic_features(df)
    else:
        result = compute_basic_features(df)
        assert "ret_1d" in result.columns
