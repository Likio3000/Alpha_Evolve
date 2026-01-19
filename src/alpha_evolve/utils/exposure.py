from __future__ import annotations

import numpy as np

_MIN_GROSS = 1e-9
_MAX_NET_TARGET = 0.95


def apply_net_exposure_target(
    signal: np.ndarray, net_target: float
) -> np.ndarray:
    """Shift weights toward a bullish net target while preserving long/short structure.

    The input signal is centered before allocating gross weight between
    long and short legs. The resulting weights have gross exposure ~= 1
    and net exposure ~= net_target (clipped to [0, _MAX_NET_TARGET]).
    """
    if signal.size == 0:
        return signal

    net = float(net_target)
    net = max(0.0, min(_MAX_NET_TARGET, net))
    centered = signal - float(np.mean(signal))

    if net <= 0.0:
        gross = float(np.sum(np.abs(centered)))
        if gross < _MIN_GROSS:
            return np.zeros_like(signal)
        return centered / gross

    pos = np.clip(centered, 0.0, None)
    neg = -np.clip(centered, None, 0.0)
    sum_pos = float(np.sum(pos))
    sum_neg = float(np.sum(neg))

    if sum_pos < _MIN_GROSS and sum_neg < _MIN_GROSS:
        return np.zeros_like(signal)
    if sum_pos < _MIN_GROSS:
        # No long signal: keep a small, uniform long bias.
        return np.full_like(signal, net / max(1, signal.size))
    if sum_neg < _MIN_GROSS:
        # No short signal: normalize longs to unit gross.
        return pos / sum_pos

    long_target = 0.5 * (1.0 + net)
    short_target = 0.5 * (1.0 - net)
    pos_scaled = pos / sum_pos * long_target
    neg_scaled = neg / sum_neg * short_target
    return pos_scaled - neg_scaled
