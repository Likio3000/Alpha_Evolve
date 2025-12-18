from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def normal_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    try:
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))
    except Exception:
        return 0.5


def trimmed_mean(values: Sequence[float] | np.ndarray, trim_frac: float) -> float:
    """Return the symmetric trimmed mean of finite values.

    ``trim_frac`` is clamped to [0, 0.499999). When 0, falls back to mean.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")

    try:
        frac = float(trim_frac)
    except Exception:
        frac = 0.0
    if not math.isfinite(frac):
        frac = 0.0
    frac = max(0.0, min(0.499999, frac))
    if frac <= 0.0:
        return float(np.mean(arr))

    xs = np.sort(arr)
    k = int(math.floor(frac * xs.size))
    if k <= 0 or 2 * k >= xs.size:
        return float(np.mean(xs))
    trimmed = xs[k : xs.size - k]
    return float(np.mean(trimmed)) if trimmed.size else float(np.mean(xs))


def safe_skew_kurtosis(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Return (skewness, kurtosis) with finite-safe defaults.

    Kurtosis is the non-excess variant (normal distribution ~= 3.0).
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return 0.0, 3.0
    mu = float(np.mean(arr))
    centered = arr - mu
    m2 = float(np.mean(centered * centered))
    if not math.isfinite(m2) or m2 < 1e-12:
        return 0.0, 3.0
    m3 = float(np.mean(centered * centered * centered))
    m4 = float(np.mean(centered * centered * centered * centered))

    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2)
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt):
        kurt = 3.0
    kurt = max(1.0, float(kurt))
    return float(skew), float(kurt)


def sharpe_std_error(sharpe: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Approximate standard error of an estimated Sharpe ratio.

    Uses the Bailey & Lopez de Prado moment-adjusted approximation:
    var(S) ~= (1 - skew*S + ((kurt-1)/4)*S^2) / (n-1)
    where kurtosis is non-excess.
    """
    try:
        n_int = int(n)
    except Exception:
        n_int = 0
    if n_int <= 1:
        return float("inf")
    s = float(sharpe)
    if not math.isfinite(s):
        return float("inf")
    try:
        g3 = float(skew)
    except Exception:
        g3 = 0.0
    if not math.isfinite(g3):
        g3 = 0.0
    try:
        g4 = float(kurt)
    except Exception:
        g4 = 3.0
    if not math.isfinite(g4):
        g4 = 3.0
    g4 = max(1.0, g4)
    denom = 1.0 - g3 * s + ((g4 - 1.0) / 4.0) * (s * s)
    if not math.isfinite(denom) or denom <= 1e-12:
        return float("inf")
    return math.sqrt(denom / float(n_int - 1))


def probabilistic_sharpe_z(
    sharpe: float,
    n: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    benchmark: float = 0.0,
) -> float:
    """Return the PSR z-score for Sharpe > benchmark."""
    se = sharpe_std_error(sharpe, n, skew=skew, kurt=kurt)
    if not math.isfinite(se) or se <= 0.0:
        return 0.0
    return (float(sharpe) - float(benchmark)) / se


def probabilistic_sharpe_ratio(
    sharpe: float,
    n: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    benchmark: float = 0.0,
) -> float:
    """Return PSR = P(true Sharpe > benchmark) under a normal approximation."""
    z = probabilistic_sharpe_z(sharpe, n, skew=skew, kurt=kurt, benchmark=benchmark)
    return float(normal_cdf(z))


def lcb_mean(mean: float, std: float, n: int, *, z: float) -> float:
    """Lower confidence bound for a mean estimate under a normal approximation."""
    mu = float(mean)
    if not math.isfinite(mu):
        return float("-inf")
    try:
        n_int = int(n)
    except Exception:
        n_int = 0
    if n_int <= 1:
        return mu
    sigma = float(std)
    if not math.isfinite(sigma) or sigma <= 1e-12:
        return mu
    z_val = float(z)
    if not math.isfinite(z_val) or z_val <= 0.0:
        return mu
    return mu - z_val * sigma / math.sqrt(float(n_int))


def prob_mean_above(
    mean: float,
    std: float,
    n: int,
    *,
    benchmark: float = 0.0,
) -> float:
    """Normal-approx probability that mean > benchmark given (mean, std, n)."""
    mu = float(mean)
    if not math.isfinite(mu):
        return 0.0
    try:
        n_int = int(n)
    except Exception:
        n_int = 0
    if n_int <= 1:
        return 0.5
    sigma = float(std)
    if not math.isfinite(sigma) or sigma <= 1e-12:
        return 1.0 if mu > benchmark else 0.0
    z = (mu - float(benchmark)) * math.sqrt(float(n_int)) / sigma
    return float(normal_cdf(z))

