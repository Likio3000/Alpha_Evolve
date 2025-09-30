#!/usr/bin/env python
"""Autonomous self-evolution driver."""

from __future__ import annotations

import argparse
import logging

from run_pipeline import PipelineOptions
from self_evolution import AgentConfig, SelfEvolutionAgent, load_search_space
from utils.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an autonomous self-evolution loop")
    p.add_argument("--config", default=None, help="Base pipeline config file (TOML)")
    p.add_argument("--search-space", required=True, help="JSON/TOML specification of tunable parameters")
    p.add_argument("--iterations", type=int, default=5, help="Number of self-evolution iterations")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--objective", default="Sharpe", help="Metric key to optimise (defaults to Sharpe)")
    p.add_argument("--minimize", action="store_true", help="Treat the objective as minimization instead of maximization")
    p.add_argument("--exploration-prob", type=float, default=0.35, help="Probability of exploring around the base configuration")
    p.add_argument("--session-root", default=None, help="Directory to store self-evolution session artifacts")
    p.add_argument("--pipeline-output-dir", default=None, help="Override pipeline output directory used for each iteration")
    p.add_argument("--no-persist-configs", action="store_true", help="Disable persistence of generated candidate config files")
    p.add_argument("--run-baselines", action="store_true", help="Train baseline models during each iteration")
    p.add_argument("--retrain-baselines", action="store_true", help="Force baseline retraining instead of using cached metrics")
    p.add_argument("--debug-prints", action="store_true", help="Enable verbose pipeline logging")
    p.add_argument("--pipeline-log-level", default=None, help="Log level passed to the pipeline (defaults to agent log level)")
    p.add_argument("--pipeline-log-file", default=None, help="Optional log file for pipeline runs")
    p.add_argument("--disable-align-cache", action="store_true", help="Disable the alignment cache for pipeline runs")
    p.add_argument("--align-cache-dir", default=None, help="Custom directory for the alignment cache")
    p.add_argument("--auto-approve", action="store_true", help="Automatically approve proposed iterations (useful for CI)")
    p.add_argument("--approval-poll-interval", type=float, default=5.0, help="Seconds between approval file checks")
    p.add_argument("--approval-timeout", type=float, default=None, help="Optional timeout (seconds) while waiting for approval")
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


def main() -> None:
    args = parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=level)

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

    pipeline_options = build_pipeline_options(args)

    agent = SelfEvolutionAgent(
        search_space=specs,
        agent_config=agent_cfg,
        base_config_path=args.config,
        pipeline_options=pipeline_options,
    )

    records = agent.run()
    best = agent.best_record
    if best and best.objective is not None:
        print(f"Best {args.objective}: {best.objective:.4f} (run: {best.run_directory})")
    else:
        print("Self-evolution run completed; no successful iterations captured")


if __name__ == "__main__":
    main()
