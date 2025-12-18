from alpha_evolve.evolution.evaluation import _compute_cv_fold_slices
from alpha_evolve.utils import stats as ae_stats


def test_cv_fold_slices_non_overlapping_and_purged():
    total_steps = 100
    k = 5
    embargo = 3
    horizon = 2

    folds = _compute_cv_fold_slices(total_steps, k, embargo=embargo, label_horizon=horizon)
    assert len(folds) == k

    for sl in folds:
        assert isinstance(sl, slice)
        assert sl.start is not None and sl.stop is not None
        assert 0 <= sl.start < sl.stop <= total_steps

    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert prev.stop <= nxt.start
        assert (nxt.start - prev.stop) >= (horizon + 2 * embargo)


def test_cv_fold_slices_zero_embargo_purges_horizon_only():
    folds = _compute_cv_fold_slices(20, 2, embargo=0, label_horizon=1)
    assert folds == [slice(0, 9), slice(10, 19)]


def test_lcb_mean_prefers_stable_signal():
    z = 1.645
    stable = ae_stats.lcb_mean(0.05, 0.05, 20, z=z)
    lucky = ae_stats.lcb_mean(0.06, 0.20, 20, z=z)
    assert stable > lucky


def test_probabilistic_sharpe_ratio_penalizes_fat_tails():
    stable = ae_stats.probabilistic_sharpe_ratio(1.0, 20, kurt=3.0)
    lucky = ae_stats.probabilistic_sharpe_ratio(1.2, 20, kurt=30.0)
    assert 0.0 <= stable <= 1.0
    assert 0.0 <= lucky <= 1.0
    assert stable > lucky


def test_safe_skew_kurtosis_defaults_for_small_samples():
    skew, kurt = ae_stats.safe_skew_kurtosis([1.0, 2.0])
    assert skew == 0.0
    assert kurt == 3.0
