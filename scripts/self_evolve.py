#!/usr/bin/env python
"""Autonomous self-evolution driver (legacy + tracked experiment modes)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha_evolve.cli.pipeline import PipelineOptions
from alpha_evolve.experiments import (
    ExperimentRegistry,
    ExperimentSessionRunner,
    ExperimentSessionSpec,
)
from alpha_evolve.self_play import (
    AgentConfig,
    SelfEvolutionAgent,
    load_base_configs,
    load_search_space,
)
from alpha_evolve.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an autonomous self-evolution loop")
    p.add_argument(
        "--mode",
        choices=("tracked", "legacy"),
        default="tracked",
        help="Execution mode: tracked (SQLite registry + dashboard approvals) or legacy (file-based approvals)",
    )
    p.add_argument("--config", default=None, help="Base pipeline config file (TOML)")
    p.add_argument(
        "--search-space",
        required=True,
        help="JSON/TOML specification of tunable parameters",
    )
    p.add_argument(
        "--iterations", type=int, default=5, help="Number of self-evolution iterations"
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--objective",
        default="Sharpe",
        help="Metric key to optimise (defaults to Sharpe)",
    )
    p.add_argument(
        "--minimize",
        action="store_true",
        help="Treat the objective as minimization instead of maximization",
    )
    p.add_argument(
        "--exploration-prob",
        type=float,
        default=0.35,
        help="Probability of exploring around the base configuration",
    )
    p.add_argument(
        "--session-root",
        default=None,
        help="Directory to store self-evolution session artifacts",
    )
    p.add_argument(
        "--registry-db",
        default=None,
        help="SQLite DB path for tracked mode (defaults to <session-root>/experiments.sqlite or artifacts/experiments/experiments.sqlite).",
    )
    p.add_argument(
        "--pipeline-output-dir",
        default=None,
        help="Override pipeline output directory used for each iteration",
    )
    p.add_argument(
        "--no-persist-configs",
        action="store_true",
        help="Disable persistence of generated candidate config files",
    )
    p.add_argument(
        "--run-baselines",
        action="store_true",
        help="Train baseline models during each iteration",
    )
    p.add_argument(
        "--retrain-baselines",
        action="store_true",
        help="Force baseline retraining instead of using cached metrics",
    )
    p.add_argument(
        "--debug-prints", action="store_true", help="Enable verbose pipeline logging"
    )
    p.add_argument(
        "--pipeline-log-level",
        default=None,
        help="Log level passed to the pipeline (defaults to agent log level)",
    )
    p.add_argument(
        "--pipeline-log-file", default=None, help="Optional log file for pipeline runs"
    )
    p.add_argument(
        "--disable-align-cache",
        action="store_true",
        help="Disable the alignment cache for pipeline runs",
    )
    p.add_argument(
        "--align-cache-dir",
        default=None,
        help="Custom directory for the alignment cache",
    )
    p.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve proposed iterations (useful for CI)",
    )
    p.add_argument(
        "--approval-poll-interval",
        type=float,
        default=5.0,
        help="Seconds between approval file checks",
    )
    p.add_argument(
        "--approval-timeout",
        type=float,
        default=None,
        help="Optional timeout (seconds) while waiting for approval",
    )
    p.add_argument(
        "--corr-gate-sharpe",
        type=float,
        default=1.0,
        help="Only consider correlation once Sharpe is above this gate (tracked mode).",
    )
    p.add_argument(
        "--sharpe-close-epsilon",
        type=float,
        default=0.05,
        help="Tie window for treating Sharpe as close (tracked mode).",
    )
    p.add_argument(
        "--max-sharpe-sacrifice",
        type=float,
        default=0.05,
        help="Max Sharpe drop allowed to accept a meaningfully lower correlation (tracked mode).",
    )
    p.add_argument(
        "--min-corr-improvement",
        type=float,
        default=0.05,
        help="Minimum improvement in avg abs corr to justify a Sharpe sacrifice (tracked mode).",
    )
    p.add_argument("--log-level", default="INFO", help="Agent log level (default INFO)")
    return p.parse_args()


def build_pipeline_options(args: argparse.Namespace) -> PipelineOptions:
    return PipelineOptions(
        debug_prints=args.debug_prints,
        run_baselines=args.run_baselines,
        retrain_baselines=args.retrain_baselines,
        log_level=args.pipeline_log_level or args.log_level,
        log_file=args.pipeline_log_file,
        output_dir=args.pipeline_output_dir,
        disable_align_cache=args.disable_align_cache,
        align_cache_dir=args.align_cache_dir,
    )


def _compute_git_sha() -> str | None:
    import subprocess

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    sha = sha.strip()
    return sha or None


def _fingerprint_dataset(path: str | None) -> str | None:
    import hashlib

    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        return None
    h = hashlib.sha256()
    if p.is_file():
        try:
            st = p.stat()
            h.update(p.name.encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
            h.update(str(int(st.st_mtime)).encode("utf-8"))
        except Exception:
            return None
        return h.hexdigest()
    files = sorted([f for f in p.glob("*.csv") if f.is_file()])
    if not files:
        return None
    for f in files:
        try:
            st = f.stat()
        except Exception:
            continue
        h.update(f.name.encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()


def main() -> None:
    args = parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=level)

    pipeline_options = build_pipeline_options(args)

    if args.mode == "legacy":
        specs = load_search_space(args.search_space)
        agent_cfg = AgentConfig(
            max_iterations=args.iterations,
            seed=args.seed,
            objective_metric=args.objective,
            maximize=not args.minimize,
            exploration_probability=args.exploration_prob,
            session_root=args.session_root,
            pipeline_output_dir=args.pipeline_output_dir,
            persist_generated_configs=not args.no_persist_configs,
            auto_approve=args.auto_approve,
            approval_poll_interval=args.approval_poll_interval,
            approval_timeout=args.approval_timeout,
        )

        agent = SelfEvolutionAgent(
            search_space=specs,
            agent_config=agent_cfg,
            base_config_path=args.config,
            pipeline_options=pipeline_options,
        )

        agent.run()
        best = agent.best_record
        if best and best.objective is not None:
            print(
                f"Best {args.objective}: {best.objective:.4f} (run: {best.run_directory})"
            )
        else:
            print("Self-evolution run completed; no successful iterations captured")
        return

    base_evo, _base_bt = load_base_configs(args.config)
    dataset_dir = base_evo.data_dir
    dataset_hash = _fingerprint_dataset(dataset_dir)
    git_sha = _compute_git_sha()

    if args.registry_db:
        db_path = Path(args.registry_db)
        if not db_path.is_absolute():
            db_path = (ROOT / db_path).resolve()
    elif args.session_root:
        db_path = Path(args.session_root) / "experiments.sqlite"
        if not db_path.is_absolute():
            db_path = (ROOT / db_path).resolve()
    else:
        db_path = (ROOT / "artifacts" / "experiments" / "experiments.sqlite").resolve()

    registry = ExperimentRegistry(db_path)
    spec = ExperimentSessionSpec(
        search_space_path=args.search_space,
        base_config_path=args.config,
        max_iterations=args.iterations,
        seed=args.seed,
        objective_metric=args.objective,
        maximize=not args.minimize,
        exploration_probability=args.exploration_prob,
        auto_approve=bool(args.auto_approve),
        approval_poll_interval=float(args.approval_poll_interval),
        approval_timeout=float(args.approval_timeout)
        if args.approval_timeout is not None
        else None,
        corr_gate_sharpe=float(args.corr_gate_sharpe),
        sharpe_close_epsilon=float(args.sharpe_close_epsilon),
        max_sharpe_sacrifice=float(args.max_sharpe_sacrifice),
        min_corr_improvement=float(args.min_corr_improvement),
        dataset_dir=str(dataset_dir),
        dataset_hash=dataset_hash,
        git_sha=git_sha,
    )
    runner = ExperimentSessionRunner(
        registry=registry,
        spec=spec,
        pipeline_options=pipeline_options,
    )
    session_id = runner.run()
    session = registry.get_session(session_id) or {}
    best_sharpe = session.get("best_sharpe")
    best_corr = session.get("best_corr")
    print(
        f"Session {session_id} completed. Best Sharpe={best_sharpe} | Best corr={best_corr} | DB={db_path}"
    )


if __name__ == "__main__":
    main()
