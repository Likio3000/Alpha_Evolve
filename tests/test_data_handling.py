import pandas as pd
import numpy as np
import pytest

from evolution_components.data_handling import _load_and_align_data_internal
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


def test_load_and_align_for_backtest_success():
    aligned, common_idx, symbols = load_and_align_data_for_backtest(
        DATA_DIR, "common_1200", 4
    )
    assert symbols == ["AAA", "BBB"]
    assert len(common_idx) == 4
    for df in aligned.values():
        assert list(df.index) == list(common_idx)


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
