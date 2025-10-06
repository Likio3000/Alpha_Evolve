import numpy as np
import pytest

from evolution_components.hall_of_fame_manager import _safe_corr


def test_safe_corr_normal():
    """Match numpy correlation when both vectors are clean and variant."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 3.0, 2.0, 5.0])
    expected = np.corrcoef(a, b)[0, 1]
    assert _safe_corr(a, b) == pytest.approx(expected)


def test_safe_corr_nan_inf():
    """Return zero when either vector contains NaN or infinite values."""
    a = np.array([1.0, np.nan, 2.0])
    b = np.array([2.0, 3.0, 4.0])
    assert _safe_corr(a, b) == 0.0
    a_inf = np.array([1.0, np.inf, 2.0])
    assert _safe_corr(a_inf, b) == 0.0


def test_safe_corr_zero_variance():
    """Return zero correlation whenever either side is constant."""
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([2.0, 3.0, 4.0])
    assert _safe_corr(a, b) == 0.0
    c = np.array([1.0, 2.0, 3.0])
    d = np.array([5.0, 5.0, 5.0])
    assert _safe_corr(c, d) == 0.0
