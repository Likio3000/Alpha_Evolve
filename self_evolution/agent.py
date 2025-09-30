from __future__ import annotations

import copy
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from config import BacktestConfig, EvolutionConfig
from run_pipeline import DEFAULT_OUTPUT_DIR, PipelineOptions, run_pipeline_programmatic
from utils.config_layering import (
    _flatten_sectioned_config,
    layer_dataclass_config,
    load_config_file,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Controls the outer self-evolution loop."""

    max_iterations: int = 5
    seed: int = 0
    objective_metric: str = "Sharpe"
    maximize: bool = True
    exploration_probability: float = 0.35
    session_root: str | None = None
    pipeline_output_dir: str | None = None
    persist_generated_configs: bool = True
    session_prefix: str = "self_evo_session"
    history_filename: str = "history.jsonl"
    summary_filename: str = "session_summary.json"
    briefing_filename: str = "agent_briefings.jsonl"
    pending_action_filename: str = "pending_action.json"
    auto_approve: bool = False
    approval_poll_interval: float = 5.0
    approval_timeout: float | None = None


@dataclass
class ParameterSpec:
    """Definition of how to sample an individual parameter."""

    key: str
    kind: str
    values: Sequence[Any] | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | None = None
    perturbation: float | None = None
    mutate_probability: float = 0.6
    allow_same: bool = False
    round_to: int | None = None
    description: str | None = None

    SKIP = object()

    def sample(
        self,
        rng: random.Random,
        *,
        base_value: Any,
        reference_value: Any | None,
        force: bool = False,
    ) -> Any:
        """Return a candidate value or `ParameterSpec.SKIP`."""

        if not force and rng.random() > self.mutate_probability:
            return self.SKIP

        value = self._generate_value(rng, base_value, reference_value)
        if self.allow_same or value != base_value:
            return value

        for _ in range(5):
            candidate = self._generate_value(rng, base_value, reference_value)
            if self.allow_same or candidate != base_value:
                return candidate
        return value

    def sample_forced(
        self,
        rng: random.Random,
        *,
        base_value: Any,
        reference_value: Any | None,
    ) -> Any:
        return self.sample(rng, base_value=base_value, reference_value=reference_value, force=True)

    def _generate_value(
        self,
        rng: random.Random,
        base_value: Any,
        reference_value: Any | None,
    ) -> Any:
        if self.kind == "choice":
            if not self.values:
                raise ValueError(f"Choice parameter '{self.key}' requires non-empty values")
            return rng.choice(list(self.values))

        if self.kind == "bool_toggle":
            current = bool(reference_value if reference_value is not None else base_value)
            return not current

        if self.kind in {"float_range", "int_range"}:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Range parameter '{self.key}' requires min_value and max_value")
            low = float(self.min_value)
            high = float(self.max_value)
            target = reference_value if reference_value is not None else base_value

            if self.perturbation and target is not None:
                center = float(target)
                delta = rng.uniform(-abs(self.perturbation), abs(self.perturbation))
                candidate = center + delta
            else:
                candidate = rng.uniform(low, high)

            candidate = max(low, min(high, candidate))

            if self.step:
                step = float(self.step)
                base = low if low != 0 else 0.0
                candidate = round((candidate - base) / step) * step + base

            if self.round_to is not None:
                candidate = round(candidate, self.round_to)

            if self.kind == "int_range":
                return int(round(candidate))
            return float(candidate)

        raise ValueError(f"Unsupported parameter kind '{self.kind}' for '{self.key}'")


@dataclass
class ParameterSpace:
    specs: Sequence[ParameterSpec]

    def sample(
        self,
        rng: random.Random,
        *,
        base_values: Mapping[str, Any],
        reference_values: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        reference_values = reference_values or {}

        for spec in self.specs:
            base_value = base_values.get(spec.key)
            ref_value = reference_values.get(spec.key, base_value)
            value = spec.sample(rng, base_value=base_value, reference_value=ref_value)
            if value is ParameterSpec.SKIP:
                continue
            updates[spec.key] = value

        if updates:
            return updates

        spec = rng.choice(list(self.specs))
        base_value = base_values.get(spec.key)
        ref_value = (reference_values or {}).get(spec.key, base_value)
        value = spec.sample_forced(rng, base_value=base_value, reference_value=ref_value)
        if value is not ParameterSpec.SKIP:
            updates[spec.key] = value
        return updates


@dataclass
class RunRecord:
    iteration: int
    run_directory: Path | None
    updates: Dict[str, Any]
    metrics: Dict[str, Any]
    objective: float | None
    success: bool
    evo_config: EvolutionConfig
    bt_config: BacktestConfig
    pipeline_options: PipelineOptions
    summary_path: Path | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "run_directory": str(self.run_directory) if self.run_directory else None,
            "updates": self.updates,
            "metrics": self.metrics,
            "objective": self.objective,
            "success": self.success,
            "summary_path": str(self.summary_path) if self.summary_path else None,
        }


def load_base_configs(config_path: str | None) -> tuple[EvolutionConfig, BacktestConfig]:
    if not config_path:
        return EvolutionConfig(), BacktestConfig()

    raw = load_config_file(config_path)
    evo_file_cfg = _flatten_sectioned_config(raw, "evolution") if "evolution" in raw else None
    bt_file_cfg = _flatten_sectioned_config(raw, "backtest") if "backtest" in raw else None
    if evo_file_cfg is None and bt_file_cfg is None:
        flat = _flatten_sectioned_config(raw, None)
        evo_file_cfg = flat
        bt_file_cfg = flat

    evo_kwargs = layer_dataclass_config(
        EvolutionConfig,
        file_cfg=evo_file_cfg,
        env_prefixes=("AE_", "AE_EVO_"),
        cli_overrides={},
    )
    bt_kwargs = layer_dataclass_config(
        BacktestConfig,
        file_cfg=bt_file_cfg,
        env_prefixes=("AE_", "AE_BT_"),
        cli_overrides={},
    )
    return EvolutionConfig(**evo_kwargs), BacktestConfig(**bt_kwargs)


def load_search_space(path: str) -> Sequence[ParameterSpec]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)

    ext = path_obj.suffix.lower()
    if ext == ".json":
        with open(path_obj, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    elif ext in {".toml", ".tml"}:  # pragma: no cover
        import tomllib

        with open(path_obj, "rb") as fh:
            data = tomllib.load(fh)
    else:
        raise RuntimeError(f"Unsupported search space file extension: {ext}")

    if isinstance(data, Mapping) and "parameters" in data:
        specs_raw = data["parameters"]
    elif isinstance(data, Mapping):
        specs_raw = [dict({"key": key}, **value) for key, value in data.items()]
    elif isinstance(data, list):
        specs_raw = data
    else:
        raise ValueError("Search space definition must be a list or mapping")

    specs: list[ParameterSpec] = []
    for item in specs_raw:
        if not isinstance(item, Mapping):
            raise ValueError("Parameter specification entries must be mappings")
        key = item.get("key")
        kind = item.get("type") or item.get("kind")
        if not key or not kind:
            raise ValueError("Each parameter spec requires 'key' and 'type'")
        spec = ParameterSpec(
            key=key,
            kind=str(kind),
            values=item.get("values"),
            min_value=item.get("min"),
            max_value=item.get("max"),
            step=item.get("step"),
            perturbation=item.get("perturbation"),
            mutate_probability=float(item.get("mutate_probability", item.get("probability", 0.6))),
            allow_same=bool(item.get("allow_same", False)),
            round_to=item.get("round_to"),
            description=item.get("description"),
        )
        specs.append(spec)

    if not specs:
        raise ValueError("Search space is empty")
    return specs


class SelfEvolutionAgent:
    def __init__(
        self,
        *,
        search_space: Sequence[ParameterSpec],
        agent_config: AgentConfig | None = None,
        base_evo_cfg: EvolutionConfig | None = None,
        base_bt_cfg: BacktestConfig | None = None,
        base_config_path: str | None = None,
        pipeline_options: PipelineOptions | None = None,
        pipeline_runner: Callable[[EvolutionConfig, BacktestConfig, PipelineOptions], Path] = run_pipeline_programmatic,
    ) -> None:
        self.agent_config = agent_config or AgentConfig()
        self.pipeline_runner = pipeline_runner
        self.pipeline_options = copy.deepcopy(pipeline_options) if pipeline_options else PipelineOptions()

        if base_evo_cfg is None or base_bt_cfg is None:
            loaded_evo, loaded_bt = load_base_configs(base_config_path)
            base_evo_cfg = base_evo_cfg or loaded_evo
            base_bt_cfg = base_bt_cfg or loaded_bt

        self.base_evo_cfg = copy.deepcopy(base_evo_cfg)
        self.base_bt_cfg = copy.deepcopy(base_bt_cfg)

        self.parameter_space = ParameterSpace(list(search_space))
        self.rng = random.Random(self.agent_config.seed)
        self.session_dir = self._initialise_session_dir()
        self.generated_configs_dir = self.session_dir / "generated_configs"
        if self.agent_config.persist_generated_configs:
            self.generated_configs_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.session_dir / self.agent_config.history_filename
        self.summary_path = self.session_dir / self.agent_config.summary_filename
        self.briefings_path = self.session_dir / self.agent_config.briefing_filename
        self.pending_action_path = self.session_dir / self.agent_config.pending_action_filename
        self.records: list[RunRecord] = []
        self.best_record: RunRecord | None = None

        if self.agent_config.pipeline_output_dir:
            self.pipeline_options.output_dir = self.agent_config.pipeline_output_dir

    def run(self) -> Sequence[RunRecord]:
        current_evo = copy.deepcopy(self.base_evo_cfg)
        current_bt = copy.deepcopy(self.base_bt_cfg)
        current_opts = copy.deepcopy(self.pipeline_options)
        current_updates: Dict[str, Any] = {}

        for iteration in range(self.agent_config.max_iterations):
            try:
                record = self._execute_pipeline_run(
                    iteration,
                    current_evo,
                    current_bt,
                    current_opts,
                    current_updates,
                )
            except Exception as exc:
                LOGGER.exception("Iteration %d failed: %s", iteration, exc)
                failure_record = RunRecord(
                    iteration=iteration,
                    run_directory=None,
                    updates=current_updates,
                    metrics={"error": str(exc)},
                    objective=None,
                    success=False,
                    evo_config=copy.deepcopy(current_evo),
                    bt_config=copy.deepcopy(current_bt),
                    pipeline_options=copy.deepcopy(current_opts),
                )
                self.records.append(failure_record)
                self._append_history(failure_record)
                self._write_pending_action(
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "iteration_completed": iteration,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                break

            prev_record = self.records[-1] if self.records else None
            analysis = self._generate_analysis(record, prev_record)

            self.records.append(record)
            self._append_history(record)
            self._maybe_update_best(record)

            is_last_iteration = iteration == self.agent_config.max_iterations - 1

            proposed_updates: Dict[str, Any] | None = None
            base_snapshot: Dict[str, Any] | None = None
            candidate_snapshot: Dict[str, Any] | None = None

            if not is_last_iteration:
                base_evo, base_bt, base_opts = self._select_base_configs()
                proposed_updates = self._sample_updates(base_evo, base_bt, base_opts)
                candidate_evo, candidate_bt, candidate_opts = self._apply_updates(
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

            code_considerations = self._suggest_code_considerations(analysis)
            self._write_briefing(record, analysis, proposed_updates, code_considerations)

            if is_last_iteration or proposed_updates is None:
                self._finalize_pending_action(record, analysis, code_considerations)
                break

            approved = self._await_approval(
                iteration=iteration,
                next_iteration=iteration + 1,
                proposed_updates=proposed_updates,
                base_snapshot=base_snapshot,
                candidate_snapshot=candidate_snapshot,
                analysis=analysis,
                code_considerations=code_considerations,
            )

            if not approved:
                LOGGER.info("Self-evolution loop halted after iteration %d", iteration)
                break

            current_evo = EvolutionConfig(**approved["evolution"])
            current_bt = BacktestConfig(**approved["backtest"])
            current_opts = PipelineOptions(**approved["pipeline_options"])
            current_updates = approved.get("updates", proposed_updates) or {}

        self._write_session_summary()
        return self.records

    def _execute_pipeline_run(
        self,
        iteration: int,
        evo_cfg: EvolutionConfig,
        bt_cfg: BacktestConfig,
        pipeline_options: PipelineOptions,
        updates: Mapping[str, Any],
    ) -> RunRecord:
        evo = copy.deepcopy(evo_cfg)
        bt = copy.deepcopy(bt_cfg)
        opts = copy.deepcopy(pipeline_options)

        if self.agent_config.persist_generated_configs:
            self._persist_candidate_config(iteration, evo, bt, opts, updates)

        run_dir = self.pipeline_runner(evo, bt, opts)
        metrics, summary_path = self._collect_metrics(run_dir)
        objective = self._extract_objective(metrics)

        LOGGER.info(
            "Iteration %d completed – objective: %s (run: %s)",
            iteration,
            objective,
            run_dir,
        )

        return RunRecord(
            iteration=iteration,
            run_directory=run_dir,
            updates=dict(updates),
            metrics=metrics,
            objective=objective,
            success=True,
            evo_config=evo,
            bt_config=bt,
            pipeline_options=opts,
            summary_path=summary_path,
        )

    def _select_base_configs(self) -> tuple[EvolutionConfig, BacktestConfig, PipelineOptions]:
        if self.best_record and self.rng.random() > self.agent_config.exploration_probability:
            LOGGER.debug("Exploiting best-known configuration")
            return (
                copy.deepcopy(self.best_record.evo_config),
                copy.deepcopy(self.best_record.bt_config),
                copy.deepcopy(self.best_record.pipeline_options),
            )
        LOGGER.debug("Exploring around base configuration")
        return (
            copy.deepcopy(self.base_evo_cfg),
            copy.deepcopy(self.base_bt_cfg),
            copy.deepcopy(self.pipeline_options),
        )

    def _sample_updates(
        self,
        evo_cfg: EvolutionConfig,
        bt_cfg: BacktestConfig,
        pipeline_options: PipelineOptions,
    ) -> Dict[str, Any]:
        base_values = self._extract_values(evo_cfg, bt_cfg, pipeline_options)
        reference_values = None
        if self.best_record:
            reference_values = self._extract_values(
                self.best_record.evo_config,
                self.best_record.bt_config,
                self.best_record.pipeline_options,
            )
        return self.parameter_space.sample(
            self.rng,
            base_values=base_values,
            reference_values=reference_values,
        )

    def _apply_updates(
        self,
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
                if not hasattr(evo_cfg, attr):
                    raise AttributeError(f"EvolutionConfig has no attribute '{attr}'")
                setattr(evo_cfg, attr, value)
            elif key.startswith("backtest."):
                attr = key.split(".", 1)[1]
                if not hasattr(bt_cfg, attr):
                    raise AttributeError(f"BacktestConfig has no attribute '{attr}'")
                setattr(bt_cfg, attr, value)
            elif key.startswith("pipeline."):
                attr = key.split(".", 1)[1]
                if not hasattr(pipeline_options, attr):
                    raise AttributeError(f"PipelineOptions has no attribute '{attr}'")
                setattr(pipeline_options, attr, value)
            else:
                raise ValueError(f"Unknown parameter prefix for '{key}'")

        return evo_cfg, bt_cfg, pipeline_options

    def _extract_values(
        self,
        evo_cfg: EvolutionConfig,
        bt_cfg: BacktestConfig,
        pipeline_options: PipelineOptions,
    ) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for spec in self.parameter_space.specs:
            if spec.key.startswith("evolution."):
                attr = spec.key.split(".", 1)[1]
                values[spec.key] = getattr(evo_cfg, attr)
            elif spec.key.startswith("backtest."):
                attr = spec.key.split(".", 1)[1]
                values[spec.key] = getattr(bt_cfg, attr)
            elif spec.key.startswith("pipeline."):
                attr = spec.key.split(".", 1)[1]
                values[spec.key] = getattr(pipeline_options, attr)
            else:
                raise ValueError(f"Unknown parameter prefix for '{spec.key}'")
        return values

    def _persist_candidate_config(
        self,
        iteration: int,
        evo_cfg: EvolutionConfig,
        bt_cfg: BacktestConfig,
        pipeline_options: PipelineOptions,
        updates: Mapping[str, Any],
    ) -> None:
        payload = {
            "iteration": iteration,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updates": dict(updates),
            "evolution": asdict(evo_cfg),
            "backtest": asdict(bt_cfg),
            "pipeline_options": asdict(pipeline_options),
        }
        out_path = self.generated_configs_dir / f"candidate_{iteration:03d}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _collect_metrics(self, run_dir: Path) -> tuple[Dict[str, Any], Path | None]:
        summary_path = run_dir / "SUMMARY.json"
        if not summary_path.exists():
            LOGGER.warning("SUMMARY.json not found in %s", run_dir)
            return {}, None
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            LOGGER.warning("Failed to read SUMMARY.json (%s): %s", summary_path, exc)
            return {}, summary_path

        metrics = dict(data.get("best_metrics", {}))
        metrics["backtested_alphas"] = data.get("backtested_alphas")
        metrics["run_dir"] = str(run_dir)
        return metrics, summary_path

    def _extract_objective(self, metrics: Mapping[str, Any]) -> float | None:
        value = metrics.get(self.agent_config.objective_metric)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _generate_analysis(self, record: RunRecord, prev_record: RunRecord | None) -> Dict[str, Any]:
        metrics = record.metrics or {}
        alpha_count = metrics.get("backtested_alphas")
        objective = record.objective
        prev_objective = None
        if prev_record and prev_record.objective is not None:
            prev_objective = prev_record.objective
        delta = None
        if objective is not None and prev_objective is not None:
            delta = objective - prev_objective

        best_before = None
        if self.best_record and self.best_record.objective is not None:
            best_before = self.best_record.objective

        improved = None
        trend = "unknown"
        if delta is not None:
            threshold = 1e-4
            if abs(delta) <= threshold:
                improved = False
                trend = "flat"
            else:
                if self.agent_config.maximize:
                    improved = delta > 0
                else:
                    improved = delta < 0
                trend = "improving" if improved else "worsening"

        plateau = False
        tail_objectives = [obj for obj in ([r.objective for r in self.records[-2:]] + [objective]) if obj is not None]
        if len(tail_objectives) >= 3:
            plateau = max(tail_objectives) - min(tail_objectives) < 1e-3

        notes: list[str] = []
        if alpha_count is None:
            notes.append("SUMMARY.json missing backtested_alphas; cannot assess alpha diversity.")
        elif alpha_count < 2:
            notes.append("Alpha output count < 2 – consider loosening diversity or increasing HoF sampling.")
        else:
            notes.append(f"Alpha output count looks healthy ({alpha_count}).")

        if delta is None:
            notes.append("No prior objective to compare – treating as baseline run.")
        else:
            if improved:
                notes.append(f"Objective improved by {delta:+.4f} vs previous run.")
            elif trend == "flat":
                notes.append("Objective change negligible vs previous run.")
            else:
                notes.append(f"Objective declined by {delta:+.4f} vs previous run.")

        if plateau:
            notes.append("Objective plateau detected over the last 3 runs.")

        if best_before is not None and objective is not None:
            rel = objective - best_before if self.agent_config.maximize else best_before - objective
            notes.append(f"Objective relative to best prior run: {rel:+.4f} (positive means improvement).")
        elif best_before is None:
            notes.append("This run establishes the initial best objective.")

        analysis = {
            "alpha_count": alpha_count,
            "objective": objective,
            "previous_objective": prev_objective,
            "objective_delta": delta,
            "trend": trend,
            "plateau": plateau,
            "improved": improved,
            "best_objective_before": best_before,
            "notes": notes,
        }
        return analysis

    def _suggest_code_considerations(self, analysis: Mapping[str, Any]) -> list[str]:
        suggestions: list[str] = []
        alpha_count = analysis.get("alpha_count")
        if isinstance(alpha_count, (int, float)) and alpha_count is not None and alpha_count < 2:
            suggestions.append(
                "Investigate Hall-of-Fame filters or correlation penalties limiting alpha diversity;"
                " consider increasing hof_per_gen or relaxing corr_cutoff."
            )

        trend = analysis.get("trend")
        if trend == "worsening":
            suggestions.append(
                "Check recent code changes around evaluation penalties (turnover, stress, factor neutrality)"
                " and verify data alignment to ensure regressions are not code-related."
            )

        if analysis.get("plateau"):
            suggestions.append(
                "Objective plateau detected – consider experimenting with new operators or mutation heuristics"
                " to boost exploration."
            )

        if not suggestions:
            suggestions.append("No immediate code changes suggested based on current metrics.")
        return suggestions

    def _write_briefing(
        self,
        record: RunRecord,
        analysis: Mapping[str, Any],
        proposed_updates: Mapping[str, Any] | None,
        code_considerations: Sequence[str],
    ) -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iteration": record.iteration,
            "run_directory": str(record.run_directory) if record.run_directory else None,
            "summary_path": str(record.summary_path) if record.summary_path else None,
            "objective_metric": self.agent_config.objective_metric,
            "metrics": record.metrics,
            "analysis": analysis,
            "proposed_updates": dict(proposed_updates) if proposed_updates else None,
            "code_considerations": list(code_considerations),
        }
        self.briefings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.briefings_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _compose_summary_text(self, analysis: Mapping[str, Any]) -> str:
        trend = analysis.get("trend", "unknown")
        objective = analysis.get("objective")
        alpha_count = analysis.get("alpha_count")
        parts = [f"Trend: {trend}"]
        if objective is not None:
            parts.append(f"Objective: {objective:.4f}")
        if isinstance(alpha_count, (int, float)) and alpha_count is not None:
            parts.append(f"Alphas: {int(alpha_count)}")
        return " | ".join(parts)

    def _await_approval(
        self,
        *,
        iteration: int,
        next_iteration: int,
        proposed_updates: Mapping[str, Any],
        base_snapshot: Mapping[str, Any] | None,
        candidate_snapshot: Mapping[str, Any] | None,
        analysis: Mapping[str, Any],
        code_considerations: Sequence[str],
    ) -> Dict[str, Any] | None:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iteration_completed": iteration,
            "next_iteration": next_iteration,
            "status": "awaiting_approval",
            "summary": self._compose_summary_text(analysis),
            "analysis": analysis,
            "proposed_updates": dict(proposed_updates),
            "proposal": {
                "base": base_snapshot,
                "candidate": candidate_snapshot,
            },
            "code_considerations": list(code_considerations),
        }

        self._write_pending_action(payload)

        if self.agent_config.auto_approve:
            auto_payload = copy.deepcopy(payload)
            auto_payload["status"] = "auto_approved"
            auto_payload["approved_updates"] = dict(proposed_updates)
            auto_payload["approved_candidate"] = candidate_snapshot
            self._write_pending_action(auto_payload)
            return {
                "evolution": candidate_snapshot["evolution"] if candidate_snapshot else base_snapshot["evolution"],
                "backtest": candidate_snapshot["backtest"] if candidate_snapshot else base_snapshot["backtest"],
                "pipeline_options": candidate_snapshot["pipeline_options"] if candidate_snapshot else base_snapshot["pipeline_options"],
                "updates": dict(proposed_updates),
            }

        LOGGER.info(
            "Waiting for approval – edit %s to approve/reject the next iteration",
            self.pending_action_path,
        )

        start = time.time()
        while True:
            if self.agent_config.approval_timeout is not None and time.time() - start > self.agent_config.approval_timeout:
                raise TimeoutError(
                    "Approval wait timed out after "
                    f"{self.agent_config.approval_timeout} seconds"
                )

            time.sleep(max(0.1, self.agent_config.approval_poll_interval))
            data = self._read_pending_action()
            if not data:
                continue
            status = str(data.get("status", "awaiting_approval")).lower()

            if status in {"approved", "auto_approved"}:
                candidate_data = data.get("approved_candidate") or data.get("proposal", {}).get("candidate") or candidate_snapshot
                updates_data = data.get("approved_updates") or data.get("proposed_updates") or proposed_updates
                if candidate_data is None:
                    candidate_data = candidate_snapshot or base_snapshot
                if candidate_data is None:
                    raise RuntimeError("Pending action missing candidate configuration on approval")
                result = {
                    "evolution": candidate_data.get("evolution", base_snapshot["evolution"] if base_snapshot else None),
                    "backtest": candidate_data.get("backtest", base_snapshot["backtest"] if base_snapshot else None),
                    "pipeline_options": candidate_data.get("pipeline_options", base_snapshot["pipeline_options"] if base_snapshot else None),
                    "updates": dict(updates_data),
                }
                data["status"] = status
                data["approved_candidate"] = candidate_data
                data["approved_updates"] = dict(updates_data)
                self._write_pending_action(data)
                return result

            if status in {"rejected", "reject", "abort", "aborted", "stop", "stopped", "terminate", "terminated"}:
                LOGGER.info("Approval file marked as %s; stopping loop.", status)
                data["status"] = status
                self._write_pending_action(data)
                return None

    def _finalize_pending_action(
        self,
        record: RunRecord,
        analysis: Mapping[str, Any],
        code_considerations: Sequence[str],
    ) -> None:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iteration_completed": record.iteration,
            "status": "session_complete",
            "summary": self._compose_summary_text(analysis),
            "latest_run": {
                "run_directory": str(record.run_directory) if record.run_directory else None,
                "summary_path": str(record.summary_path) if record.summary_path else None,
                "metrics": record.metrics,
            },
            "analysis": analysis,
            "code_considerations": list(code_considerations),
        }
        self._write_pending_action(payload)

    def _write_pending_action(self, payload: Mapping[str, Any]) -> None:
        self.pending_action_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pending_action_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _read_pending_action(self) -> Dict[str, Any] | None:
        try:
            with open(self.pending_action_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            LOGGER.warning("Pending action file is not valid JSON yet; waiting...")
            return None

    def _maybe_update_best(self, record: RunRecord) -> None:
        if not record.success:
            return
        candidate_value = record.objective
        if candidate_value is None:
            return
        if not self.best_record or self.best_record.objective is None:
            self.best_record = record
            return

        better = (
            candidate_value > self.best_record.objective
            if self.agent_config.maximize
            else candidate_value < self.best_record.objective
        )
        if better:
            self.best_record = record

    def _append_history(self, record: RunRecord) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")

    def _write_session_summary(self) -> None:
        summary = {
            "objective_metric": self.agent_config.objective_metric,
            "maximize": self.agent_config.maximize,
            "iterations": len(self.records),
            "best": self.best_record.to_dict() if self.best_record else None,
            "history_file": str(self.history_path),
            "session_dir": str(self.session_dir),
            "briefings_file": str(self.briefings_path),
            "pending_action_file": str(self.pending_action_path),
        }
        with open(self.summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    def _initialise_session_dir(self) -> Path:
        root = Path(self.agent_config.session_root or (DEFAULT_OUTPUT_DIR / "self_evolution"))
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = root / f"{self.agent_config.session_prefix}_{stamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
