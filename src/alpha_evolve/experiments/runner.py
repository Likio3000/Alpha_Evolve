from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from alpha_evolve.cli.pipeline import PipelineOptions, run_pipeline_programmatic
from alpha_evolve.config import BacktestConfig, EvolutionConfig
from alpha_evolve.experiments.registry import ExperimentRegistry
from alpha_evolve.self_play import ParameterSpace, load_base_configs, load_search_space

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentSessionSpec:
    search_space_path: str
    base_config_path: str | None = None
    max_iterations: int = 5
    seed: int = 0
    objective_metric: str = "Sharpe"
    maximize: bool = True
    exploration_probability: float = 0.35
    auto_approve: bool = False
    approval_poll_interval: float = 5.0
    approval_timeout: float | None = None

    corr_gate_sharpe: float = 1.0
    sharpe_close_epsilon: float = 0.05
    max_sharpe_sacrifice: float = 0.05
    min_corr_improvement: float = 0.05

    dataset_dir: str | None = None
    dataset_hash: str | None = None
    git_sha: str | None = None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        x = float(value)
    except Exception:
        return None
    if pd.isna(x):
        return None
    return x


def _pairwise_corr_stats(corr: pd.DataFrame | None, members: Sequence[str]) -> dict[str, Any] | None:
    if corr is None:
        return None
    members = [m for m in members if m in corr.index]
    if len(members) < 2:
        return {"k": len(members), "avg_abs_corr": 0.0, "max_abs_corr": 0.0}
    mat = corr.loc[members, members].to_numpy(dtype=float)
    mat = abs(mat)
    vals = []
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[1]):
            v = mat[i, j]
            if pd.notna(v):
                vals.append(float(v))
    if not vals:
        return {"k": len(members), "avg_abs_corr": 0.0, "max_abs_corr": 0.0}
    return {"k": int(len(members)), "avg_abs_corr": float(sum(vals) / len(vals)), "max_abs_corr": float(max(vals))}


def _collect_corr_metrics(run_dir: Path) -> tuple[float | None, float | None]:
    bt_dir = run_dir / "backtest_portfolio_csvs"
    ens_path = bt_dir / "ensemble_selection.json"
    members: list[str] = []
    if ens_path.exists():
        try:
            members = list((json.loads(ens_path.read_text(encoding="utf-8"))).get("members") or [])
        except Exception:
            members = []
    corr_path = bt_dir / "return_corr_matrix.csv"
    if not corr_path.exists() or not members:
        return None, None
    try:
        corr_df = pd.read_csv(corr_path, index_col=0)
    except Exception:
        return None, None
    stats = _pairwise_corr_stats(corr_df, members) or {}
    return _safe_float(stats.get("avg_abs_corr")), _safe_float(stats.get("max_abs_corr"))


def _extract_objective(metrics: Mapping[str, Any], objective_metric: str) -> float | None:
    value = metrics.get(objective_metric)
    if value is None:
        return None
    return _safe_float(value)


def _load_summary_metrics(run_dir: Path) -> tuple[dict[str, Any], Path | None]:
    summary_path = run_dir / "SUMMARY.json"
    if not summary_path.exists():
        return {}, None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, summary_path

    best_metrics = dict((data.get("best_metrics") or {}))
    best_metrics["backtested_alphas"] = data.get("backtested_alphas")
    return best_metrics, summary_path


def _apply_updates(
    evo_cfg: EvolutionConfig,
    bt_cfg: BacktestConfig,
    pipeline_options: PipelineOptions,
    updates: Mapping[str, Any],
) -> tuple[EvolutionConfig, BacktestConfig, PipelineOptions]:
    evo_cfg = copy.deepcopy(evo_cfg)
    bt_cfg = copy.deepcopy(bt_cfg)
    pipeline_options = copy.deepcopy(pipeline_options)

    for key, value in updates.items():
        if key.startswith("evolution."):
            attr = key.split(".", 1)[1]
            setattr(evo_cfg, attr, value)
        elif key.startswith("backtest."):
            attr = key.split(".", 1)[1]
            setattr(bt_cfg, attr, value)
        elif key.startswith("pipeline."):
            attr = key.split(".", 1)[1]
            setattr(pipeline_options, attr, value)
        else:
            raise ValueError(f"Unknown parameter prefix for '{key}'")
    return evo_cfg, bt_cfg, pipeline_options


def _is_better_multi_objective(
    *,
    candidate_sharpe: float | None,
    candidate_corr: float | None,
    best_sharpe: float | None,
    best_corr: float | None,
    maximize: bool,
    sharpe_close_epsilon: float,
    corr_gate_sharpe: float,
    max_sharpe_sacrifice: float,
    min_corr_improvement: float,
) -> bool:
    if candidate_sharpe is None:
        return False
    if best_sharpe is None:
        return True
    if not maximize:
        candidate_sharpe = -candidate_sharpe
        best_sharpe = -best_sharpe

    if candidate_sharpe > best_sharpe + sharpe_close_epsilon:
        return True
    if candidate_sharpe < best_sharpe - sharpe_close_epsilon:
        if candidate_sharpe >= corr_gate_sharpe and best_sharpe >= corr_gate_sharpe:
            sharpe_loss = best_sharpe - candidate_sharpe
            corr_best = float("inf") if best_corr is None else best_corr
            corr_cand = float("inf") if candidate_corr is None else candidate_corr
            corr_improvement = corr_best - corr_cand
            if sharpe_loss <= max_sharpe_sacrifice and corr_improvement >= min_corr_improvement:
                return True
        return False

    # Within epsilon: only let correlation influence the tie-break when both
    # candidates clear the Sharpe gate.
    if candidate_sharpe < corr_gate_sharpe or best_sharpe < corr_gate_sharpe:
        return candidate_sharpe > best_sharpe
    corr_best = float("inf") if best_corr is None else best_corr
    corr_cand = float("inf") if candidate_corr is None else candidate_corr
    return corr_cand < corr_best


class ExperimentSessionRunner:
    def __init__(
        self,
        *,
        registry: ExperimentRegistry,
        spec: ExperimentSessionSpec,
        pipeline_options: PipelineOptions | None = None,
        pipeline_runner: Callable[[EvolutionConfig, BacktestConfig, PipelineOptions], Path] = run_pipeline_programmatic,
        stop_flag: Callable[[], bool] | None = None,
    ) -> None:
        self.registry = registry
        self.spec = spec
        self.pipeline_runner = pipeline_runner
        self.stop_flag = stop_flag or (lambda: False)

        self.base_evo, self.base_bt = load_base_configs(spec.base_config_path)
        self.pipeline_options = copy.deepcopy(pipeline_options) if pipeline_options else PipelineOptions()
        self.pipeline_options.output_dir = self.pipeline_options.output_dir or None

        specs = load_search_space(spec.search_space_path)
        self.param_space = ParameterSpace(list(specs))
        import random

        self.rng = random.Random(spec.seed)
        self.best_sharpe: float | None = None
        self.best_corr: float | None = None
        self.best_configs: tuple[EvolutionConfig, BacktestConfig, PipelineOptions] | None = None

        dataset_dir = spec.dataset_dir or self.base_evo.data_dir
        self.session_id = registry.create_session(
            base_config_path=spec.base_config_path,
            search_space_path=spec.search_space_path,
            max_iterations=spec.max_iterations,
            seed=spec.seed,
            exploration_probability=spec.exploration_probability,
            objective_metric=spec.objective_metric,
            maximize=spec.maximize,
            corr_gate_sharpe=spec.corr_gate_sharpe,
            sharpe_close_epsilon=spec.sharpe_close_epsilon,
            max_sharpe_sacrifice=spec.max_sharpe_sacrifice,
            min_corr_improvement=spec.min_corr_improvement,
            dataset_dir=dataset_dir,
            dataset_hash=spec.dataset_hash,
            git_sha=spec.git_sha,
        )

        self.current_evo = copy.deepcopy(self.base_evo)
        self.current_bt = copy.deepcopy(self.base_bt)
        self.current_opts = copy.deepcopy(self.pipeline_options)
        self.current_updates: dict[str, Any] = {}

    def run(self) -> str:
        try:
            for iteration in range(int(self.spec.max_iterations)):
                if self.stop_flag():
                    self.registry.touch_session(self.session_id, status="stopped")
                    return self.session_id

                iter_id = self.registry.insert_iteration(
                    session_id=self.session_id,
                    iteration_index=iteration,
                    status="running",
                    updates=self.current_updates,
                    evolution=asdict(self.current_evo),
                    backtest=asdict(self.current_bt),
                    pipeline_options=asdict(self.current_opts),
                )

                try:
                    run_dir = self.pipeline_runner(
                        copy.deepcopy(self.current_evo),
                        copy.deepcopy(self.current_bt),
                        copy.deepcopy(self.current_opts),
                    )
                    metrics, summary_path = _load_summary_metrics(run_dir)
                    objective = _extract_objective(metrics, self.spec.objective_metric)
                    corr_avg, _corr_max = _collect_corr_metrics(run_dir)

                    self.registry.finish_iteration(
                        iteration_id=iter_id,
                        status="success",
                        run_dir=str(run_dir),
                        summary_path=str(summary_path) if summary_path else None,
                        metrics=metrics,
                        objective_sharpe=objective,
                        objective_corr=corr_avg,
                    )
                except Exception as exc:
                    self.registry.finish_iteration(
                        iteration_id=iter_id,
                        status="failed",
                        run_dir=None,
                        summary_path=None,
                        metrics={"error": str(exc)},
                        objective_sharpe=None,
                        objective_corr=None,
                    )
                    raise

                if _is_better_multi_objective(
                    candidate_sharpe=objective,
                    candidate_corr=corr_avg,
                    best_sharpe=self.best_sharpe,
                    best_corr=self.best_corr,
                    maximize=self.spec.maximize,
                    sharpe_close_epsilon=self.spec.sharpe_close_epsilon,
                    corr_gate_sharpe=self.spec.corr_gate_sharpe,
                    max_sharpe_sacrifice=self.spec.max_sharpe_sacrifice,
                    min_corr_improvement=self.spec.min_corr_improvement,
                ):
                    self.best_sharpe = objective
                    self.best_corr = corr_avg
                    self.best_configs = (
                        copy.deepcopy(self.current_evo),
                        copy.deepcopy(self.current_bt),
                        copy.deepcopy(self.current_opts),
                    )
                    self.registry.set_best(
                        self.session_id,
                        iteration_id=iter_id,
                        best_sharpe=objective,
                        best_corr=corr_avg,
                    )

                if iteration == int(self.spec.max_iterations) - 1:
                    self.registry.touch_session(self.session_id, status="completed")
                    return self.session_id

                base_evo, base_bt, base_opts = self._select_base_configs()
                proposed_updates = self._sample_updates(base_evo, base_bt, base_opts)
                candidate_evo, candidate_bt, candidate_opts = _apply_updates(
                    base_evo, base_bt, base_opts, proposed_updates
                )
                base_snapshot = {
                    "evolution": asdict(base_evo),
                    "backtest": asdict(base_bt),
                    "pipeline_options": asdict(base_opts),
                }
                candidate_snapshot = {
                    "evolution": asdict(candidate_evo),
                    "backtest": asdict(candidate_bt),
                    "pipeline_options": asdict(candidate_opts),
                }
                proposal_id = self.registry.create_proposal(
                    session_id=self.session_id,
                    iteration_completed=iteration,
                    next_iteration=iteration + 1,
                    proposed_updates=proposed_updates,
                    base_snapshot=base_snapshot,
                    candidate_snapshot=candidate_snapshot,
                )

                if self.spec.auto_approve:
                    self.registry.decide_proposal(
                        session_id=self.session_id,
                        proposal_id=proposal_id,
                        decision="approved",
                        decided_by="auto",
                        notes="auto_approve",
                    )
                    decision = "approved"
                else:
                    decision = self._await_decision(proposal_id)
                if decision == "approved":
                    self.current_evo = EvolutionConfig(**candidate_snapshot["evolution"])
                    self.current_bt = BacktestConfig(**candidate_snapshot["backtest"])
                    self.current_opts = PipelineOptions(**candidate_snapshot["pipeline_options"])
                    self.current_updates = dict(proposed_updates)
                else:
                    self.current_updates = {}
            self.registry.touch_session(self.session_id, status="completed")
            return self.session_id
        except Exception as exc:
            self.registry.touch_session(self.session_id, status="failed", last_error=str(exc))
            raise

    def _select_base_configs(self) -> tuple[EvolutionConfig, BacktestConfig, PipelineOptions]:
        if self.best_configs and self.rng.random() > float(self.spec.exploration_probability):
            evo, bt, opts = self.best_configs
            return copy.deepcopy(evo), copy.deepcopy(bt), copy.deepcopy(opts)
        return copy.deepcopy(self.base_evo), copy.deepcopy(self.base_bt), copy.deepcopy(self.pipeline_options)

    def _sample_updates(
        self,
        evo_cfg: EvolutionConfig,
        bt_cfg: BacktestConfig,
        pipeline_options: PipelineOptions,
    ) -> dict[str, Any]:
        base_values: dict[str, Any] = {}
        for spec in self.param_space.specs:
            if spec.key.startswith("evolution."):
                base_values[spec.key] = getattr(evo_cfg, spec.key.split(".", 1)[1])
            elif spec.key.startswith("backtest."):
                base_values[spec.key] = getattr(bt_cfg, spec.key.split(".", 1)[1])
            elif spec.key.startswith("pipeline."):
                base_values[spec.key] = getattr(pipeline_options, spec.key.split(".", 1)[1])
            else:
                raise ValueError(f"Unknown parameter prefix for '{spec.key}'")

        reference_values = None
        if self.best_configs:
            evo, bt, opts = self.best_configs
            reference_values = {}
            for spec in self.param_space.specs:
                if spec.key.startswith("evolution."):
                    reference_values[spec.key] = getattr(evo, spec.key.split(".", 1)[1])
                elif spec.key.startswith("backtest."):
                    reference_values[spec.key] = getattr(bt, spec.key.split(".", 1)[1])
                elif spec.key.startswith("pipeline."):
                    reference_values[spec.key] = getattr(opts, spec.key.split(".", 1)[1])
        return self.param_space.sample(self.rng, base_values=base_values, reference_values=reference_values)

    def _await_decision(self, proposal_id: int) -> str:
        start = time.time()
        while True:
            if self.stop_flag():
                self.registry.touch_session(self.session_id, status="stopped")
                return "rejected"
            if self.spec.approval_timeout is not None and (time.time() - start) > float(self.spec.approval_timeout):
                raise TimeoutError("Approval wait timed out")
            proposal = self.registry.get_proposal(self.session_id, proposal_id)
            if proposal is None:
                raise RuntimeError(f"Proposal not found: {proposal_id}")
            status = str(proposal.get("status", "pending")).lower()
            if status in {"pending"}:
                time.sleep(max(0.1, float(self.spec.approval_poll_interval)))
                continue
            return status
