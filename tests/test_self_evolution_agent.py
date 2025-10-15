import json
import random
from pathlib import Path

import pytest

from alpha_evolve.config import BacktestConfig, EvolutionConfig
from alpha_evolve.cli.pipeline import PipelineOptions
from alpha_evolve.self_play import AgentConfig, ParameterSpec, ParameterSpace, SelfEvolutionAgent


def test_parameter_spec_float_range_sampling():
    """Sample a float-range parameter and ensure values stay within bounds."""
    spec = ParameterSpec(
        key="evolution.parsimony_penalty",
        kind="float_range",
        min_value=0.001,
        max_value=0.003,
        step=0.0001,
        perturbation=0.0002,
        mutate_probability=1.0,
    )
    rng = random.Random(42)
    value = spec.sample(rng, base_value=0.002, reference_value=0.002)
    assert 0.001 <= value <= 0.003


def test_parameter_space_forces_update():
    """Check choice specs force updates even when the base value matches options."""
    spec = ParameterSpec(
        key="evolution.pop_size",
        kind="choice",
        values=[80],
        mutate_probability=0.0,
    )
    space = ParameterSpace([spec])
    updates = space.sample(random.Random(0), base_values={"evolution.pop_size": 60}, reference_values=None)
    assert updates["evolution.pop_size"] == 80


def test_self_evolution_agent_runs(tmp_path: Path):
    """Run the self-evolution agent for two iterations and verify artefacts and best record."""
    def stub_pipeline_runner(evo_cfg: EvolutionConfig, bt_cfg: BacktestConfig, options: PipelineOptions) -> Path:
        idx = stub_pipeline_runner.counter
        run_dir = tmp_path / f"run_{idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "best_metrics": {"Sharpe": 0.01 * evo_cfg.pop_size},
            "backtested_alphas": 1,
        }
        (run_dir / "SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
        stub_pipeline_runner.counter += 1
        return run_dir

    stub_pipeline_runner.counter = 0

    specs = [
        ParameterSpec(
            key="evolution.pop_size",
            kind="choice",
            values=[120],
            mutate_probability=1.0,
            allow_same=False,
        )
    ]
    agent_cfg = AgentConfig(
        max_iterations=2,
        seed=0,
        objective_metric="Sharpe",
        session_root=str(tmp_path / "sessions"),
        auto_approve=True,
        approval_poll_interval=0.01,
    )
    evo_cfg = EvolutionConfig(pop_size=80, generations=1)
    bt_cfg = BacktestConfig()
    pipeline_options = PipelineOptions(output_dir=str(tmp_path / "pipeline_runs"))

    agent = SelfEvolutionAgent(
        search_space=specs,
        agent_config=agent_cfg,
        base_evo_cfg=evo_cfg,
        base_bt_cfg=bt_cfg,
        pipeline_options=pipeline_options,
        pipeline_runner=stub_pipeline_runner,
    )

    records = agent.run()

    assert len(records) == 2
    assert agent.best_record is not None
    assert agent.best_record.objective == pytest.approx(0.01 * 120)
    generated_dir = Path(agent.session_dir) / "generated_configs"
    assert any(generated_dir.glob("candidate_*.json"))
    assert Path(agent.history_path).exists()
    briefing_lines = Path(agent.briefings_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(briefing_lines) == 2
    pending = json.loads(Path(agent.pending_action_path).read_text(encoding="utf-8"))
    assert pending.get("status") == "session_complete"
