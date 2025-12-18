from __future__ import annotations

from pathlib import Path

from alpha_evolve.experiments import ExperimentRegistry


def test_experiment_registry_session_iteration_and_proposal(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.sqlite"
    reg = ExperimentRegistry(db_path)

    session_id = reg.create_session(
        base_config_path="configs/sp500.toml",
        search_space_path="configs/self_evolution/sample_equity_space.json",
        max_iterations=2,
        seed=0,
        exploration_probability=0.25,
        objective_metric="Sharpe",
        maximize=True,
        corr_gate_sharpe=1.0,
        sharpe_close_epsilon=0.05,
        max_sharpe_sacrifice=0.05,
        min_corr_improvement=0.05,
        dataset_dir="data_sp500_small",
        dataset_hash="abc123",
        git_sha="deadbeef",
    )

    sess = reg.get_session(session_id)
    assert sess is not None
    assert sess["session_id"] == session_id
    assert sess["status"] == "running"

    iter_id = reg.insert_iteration(
        session_id=session_id,
        iteration_index=0,
        status="running",
        updates={"evolution.pop_size": 120},
        evolution={"pop_size": 120, "generations": 1},
        backtest={"top_to_backtest": 5},
        pipeline_options={"output_dir": "pipeline_runs_cs"},
    )
    reg.finish_iteration(
        iteration_id=iter_id,
        status="success",
        run_dir="pipeline_runs_cs/run_foo",
        summary_path="pipeline_runs_cs/run_foo/SUMMARY.json",
        metrics={"Sharpe": 1.2},
        objective_sharpe=1.2,
        objective_corr=0.3,
    )
    reg.set_best(session_id, iteration_id=iter_id, best_sharpe=1.2, best_corr=0.3)

    iters = reg.list_iterations(session_id)
    assert len(iters) == 1
    assert iters[0]["objective_sharpe"] == 1.2
    assert iters[0]["metrics_json"]["Sharpe"] == 1.2

    proposal_id = reg.create_proposal(
        session_id=session_id,
        iteration_completed=0,
        next_iteration=1,
        proposed_updates={"evolution.pop_size": 80},
        base_snapshot={"evolution": {"pop_size": 120}},
        candidate_snapshot={"evolution": {"pop_size": 80}},
    )
    pending = reg.get_pending_proposal(session_id)
    assert pending is not None
    assert int(pending["id"]) == proposal_id
    assert pending["status"] == "pending"

    reg.decide_proposal(session_id=session_id, proposal_id=proposal_id, decision="approved", decided_by="test")
    proposal = reg.get_proposal(session_id, proposal_id)
    assert proposal is not None
    assert proposal["status"] == "approved"

