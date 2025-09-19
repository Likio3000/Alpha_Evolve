"""Shared performance metric helpers for backtesting and evaluation components."""

from __future__ import annotations

import numpy as np


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Return the magnitude of the maximum drawdown for ``equity_curve``.

    The input is expected to be a 1-D array of cumulative equity values. The
    implementation mirrors the legacy helper previously defined inside
    ``backtesting_components.core_logic`` so callers continue to rely on the
    same behaviour while avoiding circular imports between modules.
    """

    if len(equity_curve) == 0:
        return 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-9)

    if drawdown.size == 0 or not np.any(drawdown):
        return 0.0

    # Drawdowns are negative percentages. Return the magnitude as a positive
    # number so callers don't have to negate the value.
    return float(-np.min(drawdown))


__all__ = ["compute_max_drawdown"]

