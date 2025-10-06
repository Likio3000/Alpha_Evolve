import numpy as np
import pytest

from backtesting_components.core_logic import _scale_signal_cross_sectionally, _max_drawdown


def test_scale_signal_zscore():
    """Normalize a vector with z-score scaling to produce mean-zero, unit-spread outputs."""
    vec = np.array([1.0, 2.0, 3.0])
    result = _scale_signal_cross_sectionally(vec, "zscore")
    assert result.tolist() == pytest.approx([-1.0, 0.0, 1.0], abs=1e-8)


def test_scale_signal_rank():
    """Convert raw scores into symmetric rank-based values for cross-sectional scaling."""
    vec = np.array([10.0, 20.0, 30.0])
    result = _scale_signal_cross_sectionally(vec, "rank")
    assert result.tolist() == pytest.approx([-1.0, 0.0, 1.0], abs=1e-8)


def test_scale_signal_sign():
    """Reduce signals to directional sign only, preserving zero for flat entries."""
    vec = np.array([-5.0, 0.0, 2.0])
    result = _scale_signal_cross_sectionally(vec, "sign")
    assert result.tolist() == pytest.approx([-1.0, 0.0, 1.0], abs=1e-8)


def test_max_drawdown_simple():
    """Compute the maximum drawdown from a simple equity curve to ensure downside math is correct."""
    curve = np.array([1.0, 0.8, 0.9, 0.7, 0.8])
    dd = _max_drawdown(curve)
    assert dd == pytest.approx(0.3, abs=1e-8)
