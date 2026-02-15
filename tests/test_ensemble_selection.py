import numpy as np

from alpha_evolve.backtesting.engine import (
    _corr_matrix,
    _deduplicate_results_by_return_corr,
    _select_diversified,
)


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


def test_select_diversified_strict_threshold_can_return_fewer_members():
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
        target_k=3,
        max_corr=0.1,
        corr_lambda=0.0,
        allow_relax=False,
    )
    assert selected == [0]
    assert thresholds == [0.1]


def test_select_diversified_refine_swaps_can_escape_greedy_trap():
    sharpes = [1.00, 0.99, 0.97, 0.96, 0.95]
    corr = np.array(
        [
            [1.0, 0.1, 0.2, 0.2, 0.2],
            [0.1, 1.0, 0.8, 0.8, 0.8],
            [0.2, 0.8, 1.0, 0.2, 0.2],
            [0.2, 0.8, 0.2, 1.0, 0.2],
            [0.2, 0.8, 0.2, 0.2, 1.0],
        ],
        dtype=float,
    )

    greedy_selected, _ = _select_diversified(
        sharpes,
        corr,
        target_k=3,
        max_corr=0.4,
        corr_lambda=1.0,
        relax_step=0.2,
        allow_relax=True,
        refine_swaps=False,
    )
    refined_selected, _ = _select_diversified(
        sharpes,
        corr,
        target_k=3,
        max_corr=0.4,
        corr_lambda=1.0,
        relax_step=0.2,
        allow_relax=True,
        refine_swaps=True,
    )

    assert greedy_selected == [0, 1, 2]
    assert set(refined_selected) == {0, 2, 3}
    assert 1 not in refined_selected


def test_deduplicate_results_by_return_corr_removes_near_duplicates():
    results = [
        {"AlphaID": "Alpha_01", "Sharpe": 1.2},
        {"AlphaID": "Alpha_02", "Sharpe": 1.1},
        {"AlphaID": "Alpha_03", "Sharpe": 0.9},
    ]
    per_alpha_returns = [
        ("Alpha_01", [0.01, 0.02, 0.01, 0.03]),
        # identical behaviour to Alpha_01
        ("Alpha_02", [0.02, 0.04, 0.02, 0.06]),
        ("Alpha_03", [0.01, -0.01, 0.02, -0.02]),
    ]
    kept_rows, kept_returns, report = _deduplicate_results_by_return_corr(
        results,
        per_alpha_returns,
        max_abs_corr=0.99,
    )
    kept_ids = [str(r["AlphaID"]) for r in kept_rows]
    assert kept_ids == ["Alpha_01", "Alpha_03"]
    assert [n for n, _ in kept_returns] == kept_ids
    assert report["enabled"] is True
    assert report["dropped_count"] == 1
    assert report["dropped"][0]["alpha"] == "Alpha_02"


def test_deduplicate_results_by_return_corr_disables_when_threshold_out_of_range():
    results = [
        {"AlphaID": "Alpha_01", "Sharpe": 1.2},
        {"AlphaID": "Alpha_02", "Sharpe": 1.1},
    ]
    per_alpha_returns = [
        ("Alpha_01", [0.01, 0.02, 0.01, 0.03]),
        ("Alpha_02", [0.02, 0.04, 0.02, 0.06]),
    ]
    kept_rows, kept_returns, report = _deduplicate_results_by_return_corr(
        results,
        per_alpha_returns,
        max_abs_corr=1.0,
    )
    assert [str(r["AlphaID"]) for r in kept_rows] == ["Alpha_01", "Alpha_02"]
    assert [n for n, _ in kept_returns] == ["Alpha_01", "Alpha_02"]
    assert report["enabled"] is False
