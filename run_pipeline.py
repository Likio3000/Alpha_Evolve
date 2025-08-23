#!/usr/bin/env python
"""
run_pipeline.py  – evolve cross-sectional alphas **and** back-test them.
Usage example:
    uv run run_pipeline.py 5 --max_lookback_data_option full_overlap --fee 0.5 --debug_prints

Operation limit flags:
    --max_setup_ops
    --max_predict_ops
    --max_update_ops
    --max_scalar_operands
    --max_vector_operands
    --max_matrix_operands
    --eval_cache_size
"""

from __future__ import annotations
import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

from config import EvolutionConfig, BacktestConfig
import evolve_alphas as ae
import backtest_evolved_alphas as bt
from utils.logging_setup import setup_logging

BASE_OUTPUT_DIR = Path("./pipeline_runs_cs")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI → two dataclass configs
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> tuple[EvolutionConfig, BacktestConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description="Evolve and back-test alphas (one-stop shop)")

    # ───► evolution flags
    p.add_argument("generations", type=int)
    p.add_argument("--seed",               type=int,   default=argparse.SUPPRESS)
    p.add_argument("--pop_size",           type=int,   default=argparse.SUPPRESS)
    p.add_argument("--tournament_k",       type=int,   default=argparse.SUPPRESS)
    p.add_argument("--p_mut",              type=float, default=argparse.SUPPRESS)
    p.add_argument("--p_cross",            type=float, default=argparse.SUPPRESS)
    p.add_argument("--elite_keep",         type=int,   default=argparse.SUPPRESS)
    p.add_argument("--fresh_rate",         type=float, default=argparse.SUPPRESS)
    p.add_argument("--max_ops",            type=int,   default=argparse.SUPPRESS)
    p.add_argument("--max_setup_ops",      type=int,   default=argparse.SUPPRESS)
    p.add_argument("--max_predict_ops",    type=int,   default=argparse.SUPPRESS)
    p.add_argument("--max_update_ops",     type=int,   default=argparse.SUPPRESS)
    p.add_argument("--max_scalar_operands", type=int,  default=argparse.SUPPRESS)
    p.add_argument("--max_vector_operands", type=int,  default=argparse.SUPPRESS)
    p.add_argument("--max_matrix_operands", type=int,  default=argparse.SUPPRESS)
    p.add_argument("--parsimony_penalty",  type=float, default=argparse.SUPPRESS)
    p.add_argument("--corr_penalty_w",     type=float, default=argparse.SUPPRESS)
    p.add_argument("--corr_cutoff",        type=float, default=argparse.SUPPRESS)
    p.add_argument("--sharpe_proxy_w",     type=float, default=argparse.SUPPRESS)
    p.add_argument("--ic_std_penalty_w",   type=float, default=argparse.SUPPRESS)
    p.add_argument("--turnover_penalty_w", type=float, default=argparse.SUPPRESS)
    p.add_argument("--use_train_val_splits", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--train_points",       type=int,   default=argparse.SUPPRESS)
    p.add_argument("--val_points",         type=int,   default=argparse.SUPPRESS)
    p.add_argument("--keep_dupes_in_hof", action="store_true",
                   default=argparse.SUPPRESS)
    p.add_argument("--xs_flat_guard",      type=float, default=argparse.SUPPRESS)
    p.add_argument("--t_flat_guard",       type=float, default=argparse.SUPPRESS)
    p.add_argument("--early_abort_bars",   type=int,   default=argparse.SUPPRESS)
    p.add_argument("--early_abort_xs",     type=float, default=argparse.SUPPRESS)
    p.add_argument("--early_abort_t",      type=float, default=argparse.SUPPRESS)
    p.add_argument("--flat_bar_threshold", type=float, default=argparse.SUPPRESS)
    p.add_argument("--hof_size",           type=int,   default=argparse.SUPPRESS)
    p.add_argument("--scale",              choices=["zscore","rank","sign","madz","winsor"],
                                                     default=argparse.SUPPRESS)
    p.add_argument("--sector_neutralize",  action="store_true", default=argparse.SUPPRESS,
                   help="Demean signals by sector prior to IC")
    p.add_argument("--winsor_p",           type=float, default=argparse.SUPPRESS,
                   help="Tail probability for 'winsor' scaling")
    p.add_argument("--quiet",              action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--workers",            type=int,   default=argparse.SUPPRESS)
    p.add_argument("--eval_cache_size",    type=int,   default=argparse.SUPPRESS)

    # ───► shared data flags
    p.add_argument("--data_dir",                 default=argparse.SUPPRESS)
    p.add_argument("--max_lookback_data_option", choices=['common_1200',
                                                          'specific_long_10k',
                                                          'full_overlap'],
                                                   default=argparse.SUPPRESS)
    p.add_argument("--min_common_points",  type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval_lag",           type=int, default=argparse.SUPPRESS)

    # ───► back-test-only flags
    p.add_argument("--top",   dest="top_to_backtest", type=int, default=argparse.SUPPRESS)
    p.add_argument("--top_to_backtest", dest="top_to_backtest", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    p.add_argument("--fee",                            type=float, default=argparse.SUPPRESS)
    p.add_argument("--hold",                           type=int,   default=argparse.SUPPRESS)
    p.add_argument("--long_short_n",                   type=int,   default=argparse.SUPPRESS)
    p.add_argument("--annualization_factor", type=float, default=argparse.SUPPRESS)
    p.add_argument("--debug_prints", action="store_true")
    p.add_argument("--run_baselines", action="store_true",
                   help="also train baseline models")
    p.add_argument("--retrain_baselines", action="store_true",
                   help="force training even if cached metrics exist")
    p.add_argument("--log-level", default="INFO",
                   help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    p.add_argument("--log-file", default=None,
                   help="File to additionally write logs to")
    p.add_argument("--dry-run", action="store_true",
                   help="Show resolved configs and planned outputs, then exit")

    ns = p.parse_args()
    d = vars(ns)

    evo_cfg = EvolutionConfig(**{k: v for k, v in d.items()
                                 if k in EvolutionConfig.__annotations__})
    bt_cfg  = BacktestConfig(**{k: v for k, v in d.items()
                                 if k in BacktestConfig.__annotations__})

    return evo_cfg, bt_cfg, ns


# ─────────────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────────────
def _evolve_and_save(cfg: EvolutionConfig, run_output_dir: Path) -> Path:
    import time
    logger = logging.getLogger(__name__)
    logger.info(f"\n— Evolution: {cfg.generations} generations  (seed {cfg.seed})")
    hof = ae.evolve(cfg)                                   # List[(AlphaProgram, IC)]
    hof = hof[:cfg.hof_size]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = (run_output_dir / "pickles" /
                f"evolved_top{cfg.hof_size}_{cfg.generations}g_"
                f"{cfg.max_lookback_data_option}_{stamp}.pkl")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as fh:
        pickle.dump(hof, fh)

    logger.info(f"Saved {len(hof)} programmes → {out_file}")
    return out_file


def _train_baselines(data_dir: str, out_dir: Path, *, retrain: bool = False) -> None:
    """Train baseline models and dump their metrics as JSON.

    If a ``baseline_metrics.json`` file already exists inside ``data_dir`` and
    ``retrain`` is ``False`` the metrics are loaded instead of re-training the
    models.  Metrics are always written to ``out_dir``.
    """
    import json
    from baselines.ga_tree import train_ga_tree
    from baselines.rank_lstm import train_rank_lstm

    cache_path = Path(data_dir) / "baseline_metrics.json"

    logger = logging.getLogger(__name__)

    if not retrain and cache_path.exists():
        with open(cache_path) as fh:
            metrics = json.load(fh)
        logger.info(f"Loaded baseline metrics from cache → {cache_path}")
    else:
        metrics = {
            "ga_tree": train_ga_tree(data_dir),
            "rank_lstm": train_rank_lstm(data_dir),
        }
        with open(cache_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info(f"Trained baseline models and cached metrics → {cache_path}")

    name_map = {
        "ga_tree": "GA tree",
        "rank_lstm": "RankLSTM",
    }
    for name, m in metrics.items():
        nm = name_map.get(name, name)
        logger.info(f"{nm} IC: {m['IC']:.4f} Sharpe: {m['Sharpe']:.4f}")

    out_file = out_dir / "baseline_metrics.json"
    with open(out_file, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Saved baseline metrics → {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    evo_cfg, bt_cfg, cli = parse_args()

    level = getattr(logging, str(cli.log_level).upper(), logging.INFO)
    if getattr(cli, "debug_prints", False):
        level = logging.DEBUG
    elif getattr(cli, "quiet", False):
        level = logging.WARNING

    setup_logging(level=level, log_file=cli.log_file)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (BASE_OUTPUT_DIR /
               f"run_g{evo_cfg.generations}_seed{evo_cfg.seed}_"
               f"{evo_cfg.max_lookback_data_option}_{run_stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist configs and minimal run metadata for reproducibility
    try:
        import json, platform, subprocess
        from dataclasses import asdict
        (run_dir / "meta").mkdir(exist_ok=True)
        with open(run_dir / "meta" / "evolution_config.json", "w") as fh:
            json.dump(asdict(evo_cfg), fh, indent=2)
        with open(run_dir / "meta" / "backtest_config.json", "w") as fh:
            json.dump(asdict(bt_cfg), fh, indent=2)
        meta = {
            "python": sys.version,
            "platform": platform.platform(),
            "time": run_stamp,
        }
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            meta["git_commit"] = sha
        except Exception:
            pass
        with open(run_dir / "meta" / "run_metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        try:
            with open(BASE_OUTPUT_DIR / "LATEST", "w") as fh:
                fh.write(str(run_dir))
        except Exception:
            pass
    except Exception:
        logging.getLogger(__name__).warning("Failed to save run metadata.")

    # Dry-run: show the plan, paths, and exit early
    if getattr(cli, "dry_run", False):
        logging.getLogger(__name__).info(
            "Dry run: would evolve %d generations and backtest top %d; outputs → %s",
            evo_cfg.generations,
            bt_cfg.top_to_backtest,
            run_dir,
        )
        # Write a brief README
        try:
            with open(run_dir / "README.txt", "w") as fh:
                fh.write(
                    f"Alpha Evolve Pipeline (dry run)\n\n"
                    f"Run directory: {run_dir}\n"
                    f"Generations: {evo_cfg.generations}\n"
                    f"Data: {bt_cfg.data_dir}\n"
                    f"Backtest top: {bt_cfg.top_to_backtest}\n"
                )
        except Exception:
            pass
        return

    pickle_path = _evolve_and_save(evo_cfg, run_dir)

    # Save per-generation diagnostics if available
    try:
        from evolution_components import diagnostics as diag
        diags = diag.get_all()
        if diags:
            import json
            diag_path = run_dir / "diagnostics.json"
            with open(diag_path, "w") as fh:
                json.dump(diags, fh, indent=2)
            logging.getLogger(__name__).info(f"Saved diagnostics → {diag_path}")
    except Exception:
        pass

    # Back-test programmatically using the new API
    logger = logging.getLogger(__name__)
    logger.info("\n— Back-testing …")
    try:
        bt.run(
            bt_cfg,
            outdir=run_dir / "backtest_portfolio_csvs",
            programs_pickle=pickle_path,
            debug_prints=getattr(cli, "debug_prints", False),
            annualization_factor_override=None,
            logger=logger,
        )
    except Exception as e:
        logger.exception("Back-testing failed: %s", e)
        sys.exit(1)

    if cli.run_baselines:
        _train_baselines(bt_cfg.data_dir, run_dir,
                         retrain=getattr(cli, "retrain_baselines", False))

    logger.info(f"\n✔  Pipeline finished – artefacts in  {run_dir}")


if __name__ == "__main__":
    main()
