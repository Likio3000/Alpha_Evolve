import numpy as np

from alpha_evolve.backtesting.engine import _corr_matrix, _select_diversified


def test_corr_matrix_is_finite_for_constant_series():
    R = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    corr = _corr_matrix(R)
    assert corr.shape == (2, 2)
    assert np.isfinite(corr).all()
    assert corr[0, 1] == 0.0
    assert corr[1, 0] == 0.0


def test_select_diversified_prefers_uncorrelated_with_soft_penalty():
    sharpes = [1.0, 0.99, 0.98]
    corr = np.array(
        [
            [1.0, 0.95, 0.0],
            [0.95, 1.0, 0.1],
            [0.0, 0.1, 1.0],
        ],
        dtype=float,
    )
    selected, thresholds = _select_diversified(
        sharpes,
        corr,
        target_k=2,
        max_corr=0.999,
        corr_lambda=1.0,
    )
    assert selected == [0, 2]
    assert len(thresholds) == 2


def test_select_diversified_relaxes_threshold_when_needed():
    sharpes = [1.0, 0.99, 0.98]
    corr = np.array(
        [
            [1.0, 0.99, 0.99],
            [0.99, 1.0, 0.99],
            [0.99, 0.99, 1.0],
        ],
        dtype=float,
    )
    selected, thresholds = _select_diversified(
        sharpes,
        corr,
        target_k=2,
        max_corr=0.1,
        corr_lambda=0.0,
        relax_step=0.2,
    )
    assert selected[0] == 0
    assert len(selected) == 2
    assert thresholds[1] > 0.1

