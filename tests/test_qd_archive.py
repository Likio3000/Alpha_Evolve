from __future__ import annotations


from alpha_evolve.evolution import qd_archive
from alpha_evolve.evolution.evaluation import EvalResult


class DummyProgram:
    def __init__(self, *, size: int, fingerprint: str) -> None:
        self.fingerprint = fingerprint
        self.size = size


def _dummy_program(size: int, fingerprint: str) -> DummyProgram:
    return DummyProgram(size=size, fingerprint=fingerprint)


def _eval_result(fitness: float, turnover: float) -> EvalResult:
    return EvalResult(
        fitness=fitness,
        mean_ic=0.0,
        sharpe_proxy=0.0,
        parsimony_penalty=0.0,
        correlation_penalty=0.0,
        processed_predictions=None,
        ic_std=0.0,
        turnover_proxy=turnover,
        factor_penalty=0.0,
        fitness_static=None,
    )


def test_qd_archive_keeps_best_per_cell():
    """Ensure the QD archive retains only the top candidate per discretized cell."""
    qd_archive.initialize_archive(
        turnover_bins=(0.1, 0.3), complexity_bins=(0.5,), max_entries=10
    )

    prog_a = _dummy_program(size=10, fingerprint="A")
    prog_b = _dummy_program(size=40, fingerprint="B")
    prog_c = _dummy_program(size=10, fingerprint="C")

    res_a = _eval_result(fitness=0.1, turnover=0.05)
    res_b = _eval_result(fitness=0.2, turnover=0.4)
    res_c = _eval_result(fitness=0.3, turnover=0.05)

    qd_archive.add_candidate(prog=prog_a, metrics=res_a, generation=0, max_ops=100)
    qd_archive.add_candidate(prog=prog_b, metrics=res_b, generation=0, max_ops=100)
    # Same cell as A but higher fitness should replace
    qd_archive.add_candidate(prog=prog_c, metrics=res_c, generation=1, max_ops=100)

    elites = qd_archive.get_elites()
    fingerprints = {entry.fingerprint for entry in elites}
    assert fingerprints == {"B", "C"}
    summary = qd_archive.get_summary()
    assert summary["cells"] == 2
    assert any(elite["fingerprint"] == "C" for elite in summary["elites"])

    qd_archive.clear_archive()
    assert qd_archive.get_elites() == []
