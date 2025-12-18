from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_evolve.cli.pipeline import PipelineOptions
from alpha_evolve.config import BacktestConfig, EvolutionConfig
from alpha_evolve.experiments import ExperimentRegistry, ExperimentSessionRunner, ExperimentSessionSpec


def _write_corr_artifacts(run_dir: Path, *, members: list[str], corr: list[list[float]]) -> None:
    bt_dir = run_dir / "backtest_portfolio_csvs"
    bt_dir.mkdir(parents=True, exist_ok=True)
    (bt_dir / "ensemble_selection.json").write_text(json.dumps({"members": members}), encoding="utf-8")
    df = pd.DataFrame(corr, index=members, columns=members)
    df.to_csv(bt_dir / "return_corr_matrix.csv")


def test_experiment_session_runner_records_and_auto_approves(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.sqlite"
    reg = ExperimentRegistry(db_path)

    space_path = tmp_path / "space.json"
    space_path.write_text(
        json.dumps(
            {
                "parameters": [
                    {"key": "evolution.pop_size", "type": "choice", "values": [120], "mutate_probability": 1.0}
                ]
            }
        ),
        encoding="utf-8",
    )

    def stub_pipeline_runner(evo_cfg: EvolutionConfig, bt_cfg: BacktestConfig, options: PipelineOptions) -> Path:
        idx = stub_pipeline_runner.counter
        run_dir = tmp_path / f"run_{idx}"
        run_dir.mkdir(parents=True, exist_ok=True)

        sharpe = 0.01 * float(evo_cfg.pop_size)
        summary = {"best_metrics": {"Sharpe": sharpe, "AlphaID": f"Alpha_{idx:02d}"}, "backtested_alphas": 3}
        (run_dir / "SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
        _write_corr_artifacts(run_dir, members=["Alpha_01", "Alpha_02", "Alpha_03"], corr=[[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]])

        stub_pipeline_runner.counter += 1
        return run_dir

    stub_pipeline_runner.counter = 0

    spec = ExperimentSessionSpec(
        search_space_path=str(space_path),
        base_config_path=None,
        max_iterations=2,
        seed=0,
        objective_metric="Sharpe",
        maximize=True,
        exploration_probability=0.0,
        auto_approve=True,
        approval_poll_interval=0.01,
        corr_gate_sharpe=0.5,
        sharpe_close_epsilon=0.05,
        max_sharpe_sacrifice=0.05,
        min_corr_improvement=0.05,
    )

    runner = ExperimentSessionRunner(
        registry=reg,
        spec=spec,
        pipeline_options=PipelineOptions(output_dir=str(tmp_path)),
        pipeline_runner=stub_pipeline_runner,
    )
    session_id = runner.run()

    sess = reg.get_session(session_id)
    assert sess is not None
    assert sess["status"] == "completed"
    assert sess["best_iteration_id"] is not None
    assert sess["best_sharpe"] == pytest.approx(1.2)

    iters = reg.list_iterations(session_id)
    assert len(iters) == 2
    assert iters[0]["objective_sharpe"] == pytest.approx(1.0)
    assert iters[1]["objective_sharpe"] == pytest.approx(1.2)

    proposals = reg.list_proposals(session_id)
    assert proposals, "Expected an auto-approved proposal for the second iteration."
    assert proposals[0]["status"] == "approved"
