"""Utilities for multi-objective evolutionary selection (NSGA-II style)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math

from evolution_components.evaluation_logic import EvalResult

ObjectiveTuple = Tuple[float, ...]
OBJECTIVE_LABELS: Tuple[str, ...] = (
    "ic",
    "sh",
    "neg_turn",
    "neg_complex",
    "neg_dd",
    "neg_factor",
    "neg_robust",
)


@dataclass
class ParetoAnalysis:
    """Summary of Pareto fronts for a population."""

    objectives: Dict[int, ObjectiveTuple] = field(default_factory=dict)
    fronts: List[List[int]] = field(default_factory=list)
    ranks: Dict[int, int] = field(default_factory=dict)
    crowding: Dict[int, float] = field(default_factory=dict)
    scores: Dict[int, float] = field(default_factory=dict)
    labels: Tuple[str, ...] = OBJECTIVE_LABELS

    def front(self, rank: int) -> List[int]:
        return self.fronts[rank] if 0 <= rank < len(self.fronts) else []


def default_objectives(result: EvalResult) -> ObjectiveTuple:
    """Project evaluation metrics into a maximisation objective tuple."""

    return (
        float(result.mean_ic),
        float(result.sharpe_proxy),
        float(-result.turnover_proxy),
        float(-result.parsimony_penalty),
        float(-getattr(result, "max_drawdown", 0.0)),
        float(-getattr(result, "factor_exposure_sum", 0.0)),
        float(-getattr(result, "robustness_penalty", 0.0)),
    )


def dominates(a: ObjectiveTuple, b: ObjectiveTuple) -> bool:
    """Return True if ``a`` Pareto-dominates ``b`` (all >=, and strictly > in one)."""

    ge = True
    gt = False
    for av, bv in zip(a, b):
        if av < bv:
            ge = False
            break
        if av > bv:
            gt = True
    return ge and gt


def nondominated_sort(objs: Sequence[ObjectiveTuple]) -> List[List[int]]:
    """Fast non-dominated sorting (Deb et al., NSGA-II).

    Returns indices (relative to ``objs``) grouped by Pareto front rank.
    """

    n = len(objs)
    S: List[set[int]] = [set() for _ in range(n)]
    dominance_count = [0] * n
    fronts: List[List[int]] = []

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].add(q)
            elif dominates(objs[q], objs[p]):
                dominance_count[p] += 1
        if dominance_count[p] == 0:
            if not fronts:
                fronts.append([])
            fronts[0].append(p)

    i = 0
    while i < len(fronts):
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                dominance_count[q] -= 1
                if dominance_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def crowding_distance(front: Sequence[int], objs: Sequence[ObjectiveTuple]) -> Dict[int, float]:
    """Compute crowding distance for indices in ``front``.

    Distances follow NSGA-II: boundary points receive ``inf``.
    """

    if not front:
        return {}
    m = len(objs[0]) if objs else 0
    distance = {idx: 0.0 for idx in front}
    if m == 0:
        return distance
    for k in range(m):
        front_sorted = sorted(front, key=lambda idx: objs[idx][k])
        distance[front_sorted[0]] = math.inf
        distance[front_sorted[-1]] = math.inf
        min_val = objs[front_sorted[0]][k]
        max_val = objs[front_sorted[-1]][k]
        span = max_val - min_val
        if not math.isfinite(span) or span <= 0.0:
            continue
        for j in range(1, len(front_sorted) - 1):
            prev_val = objs[front_sorted[j - 1]][k]
            next_val = objs[front_sorted[j + 1]][k]
            distance[front_sorted[j]] += (next_val - prev_val) / span
    return distance


def compute_pareto_analysis(
    eval_results: Sequence[Tuple[int, EvalResult]],
    objective_fn: Optional[Callable[[EvalResult], ObjectiveTuple]] = None,
) -> ParetoAnalysis:
    """Build Pareto fronts, ranks, and scalar scores for selection."""

    objective_fn = objective_fn or default_objectives
    valid_entries: List[Tuple[int, ObjectiveTuple]] = []
    for pop_idx, res in eval_results:
        if not math.isfinite(res.fitness):
            continue
        obj = objective_fn(res)
        if any(math.isnan(v) for v in obj):
            continue
        valid_entries.append((pop_idx, obj))

    if not valid_entries:
        return ParetoAnalysis()

    indices = [idx for idx, _ in valid_entries]
    objs = [obj for _, obj in valid_entries]
    fronts_compact = nondominated_sort(objs)
    fronts: List[List[int]] = []
    ranks: Dict[int, int] = {}
    crowding: Dict[int, float] = {}

    for rank, front in enumerate(fronts_compact):
        if not front:
            continue
        front_pop = [indices[i] for i in front]
        fronts.append(front_pop)
        front_crowding_compact = crowding_distance(front, objs)
        for rel_idx, pop_idx in zip(front, front_pop):
            ranks[pop_idx] = rank
            crowd_val = front_crowding_compact.get(rel_idx, 0.0)
            crowding[pop_idx] = crowd_val

    # Provide fallback scores that respect rank (higher is better) with crowding tie-breaker.
    scores: Dict[int, float] = {}
    for pop_idx, obj in valid_entries:
        rank = ranks.get(pop_idx)
        crowd_val = crowding.get(pop_idx, 0.0)
        if rank is None:
            continue
        adj_crowd = crowd_val if math.isfinite(crowd_val) else 1e6
        scores[pop_idx] = -float(rank) + 1e-3 * adj_crowd

    objectives_map = {idx: obj for idx, obj in valid_entries}
    return ParetoAnalysis(
        objectives=objectives_map,
        fronts=fronts,
        ranks=ranks,
        crowding=crowding,
        scores=scores,
    )


def to_objective_dict(obj: ObjectiveTuple, labels: Sequence[str] = OBJECTIVE_LABELS) -> Dict[str, float]:
    """Utility to map an objective tuple into a dict keyed by ``labels``."""

    return {label: float(value) for label, value in zip(labels, obj)}
