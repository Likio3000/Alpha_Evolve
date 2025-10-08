import pytest

from evolution_components import moea
from evolution_components.evaluation_logic import EvalResult


def make_result(*, fitness: float, mean_ic: float, sharpe: float, turnover: float, parsimony: float,
                drawdown: float, factor_sum: float, robustness: float,
                sortino: float = 0.0, downside: float = 0.0, cvar: float = -0.01) -> EvalResult:
    return EvalResult(
        fitness=fitness,
        mean_ic=mean_ic,
        sharpe_proxy=sharpe,
        parsimony_penalty=parsimony,
        correlation_penalty=0.0,
        processed_predictions=None,
        turnover_proxy=turnover,
        max_drawdown=drawdown,
        factor_exposure_sum=factor_sum,
        robustness_penalty=robustness,
        sortino_ratio=sortino,
        downside_deviation=downside,
        cvar=cvar,
    )


def test_default_objectives_negate_penalties():
    """Ensure default MOEA objective vector keeps rewards positive and penalties negated."""
    res = make_result(
        fitness=0.4,
        mean_ic=0.05,
        sharpe=1.1,
        turnover=0.2,
        parsimony=0.08,
        drawdown=0.03,
        factor_sum=0.11,
        robustness=0.07,
        sortino=1.4,
        downside=0.05,
        cvar=-0.025,
    )
    objectives = moea.default_objectives(res)
    assert objectives[0] == pytest.approx(res.mean_ic)
    assert objectives[1] == pytest.approx(res.sharpe_proxy)
    assert objectives[2] == pytest.approx(res.sortino_ratio)
    assert objectives[3] == pytest.approx(-res.turnover_proxy)
    assert objectives[4] == pytest.approx(-res.parsimony_penalty)
    assert objectives[5] == pytest.approx(-res.max_drawdown)
    assert objectives[6] == pytest.approx(-res.factor_exposure_sum)
    assert objectives[7] == pytest.approx(-res.robustness_penalty)
    assert objectives[8] == pytest.approx(-max(0.0, -res.cvar))


def test_compute_pareto_analysis_penalises_turnover_drawdown_and_factor_exposure():
    """Check Pareto analysis ranks penalized candidates lower while retaining balanced trade-offs."""
    dominant = make_result(
        fitness=0.6,
        mean_ic=0.05,
        sharpe=1.05,
        turnover=0.12,
        parsimony=0.06,
        drawdown=0.03,
        factor_sum=0.05,
        robustness=0.02,
        sortino=1.2,
        cvar=-0.03,
    )
    dominated = make_result(
        fitness=0.6,
        mean_ic=0.05,
        sharpe=1.05,
        turnover=0.4,
        parsimony=0.08,
        drawdown=0.09,
        factor_sum=0.2,
        robustness=0.06,
        sortino=0.8,
        cvar=-0.06,
    )
    tradeoff = make_result(
        fitness=0.45,
        mean_ic=0.04,
        sharpe=0.9,
        turnover=0.05,
        parsimony=0.05,
        drawdown=0.02,
        factor_sum=0.01,
        robustness=0.01,
        sortino=1.0,
        cvar=-0.015,
    )
    analysis = moea.compute_pareto_analysis([
        (0, dominant),
        (1, dominated),
        (2, tradeoff),
    ])

    assert analysis.ranks[0] == 0
    assert analysis.ranks[1] > analysis.ranks[0]
    assert 1 not in analysis.front(0)
    assert analysis.scores[0] > analysis.scores[1]

    # Trade-off candidate should survive on the first front because of lower penalties
    assert 2 in analysis.front(0)

    obj_dict = moea.to_objective_dict(analysis.objectives[0])
    assert obj_dict["neg_turn"] == pytest.approx(-dominant.turnover_proxy)
    assert obj_dict["sortino"] == pytest.approx(dominant.sortino_ratio)
    assert obj_dict["neg_cvar"] == pytest.approx(-max(0.0, -dominant.cvar))
    assert set(obj_dict.keys()) == set(moea.OBJECTIVE_LABELS)
