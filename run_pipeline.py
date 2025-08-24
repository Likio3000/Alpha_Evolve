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
from dataclasses import fields as dc_fields

from config import EvolutionConfig, BacktestConfig
import evolve_alphas as ae
import backtest_evolved_alphas as bt
from utils.logging_setup import setup_logging

BASE_OUTPUT_DIR = Path("./pipeline_runs_cs")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI → two dataclass configs (auto-generated from dataclasses)
# ─────────────────────────────────────────────────────────────────────────────
def _add_dataclass_args(parser: argparse.ArgumentParser,
                        dc_type,
                        *,
                        skip: set[str] | None = None,
                        choices_map: dict[str, list[str]] | None = None,
                        already_added: set[str] | None = None) -> set[str]:
    skip = skip or set()
    choices_map = choices_map or {}
    added = set() if already_added is None else set(already_added)
    for f in dc_fields(dc_type):
        name = f.name
        if name in skip or name in added:
            continue
        ftype = f.type
        if ftype not in (int, float, str, bool):
            continue
        if name == "generations":
            # Keep positional form for compatibility
            continue
        arg = f"--{name}"
        kwargs: dict = {"default": argparse.SUPPRESS}
        if ftype is bool:
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = ftype
            if name in choices_map:
                kwargs["choices"] = choices_map[name]
        parser.add_argument(arg, **kwargs)
        added.add(name)
    return added


def parse_args() -> tuple[EvolutionConfig, BacktestConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description="Evolve and back-test alphas (one-stop shop)")

    # Positional generations
    p.add_argument("generations", type=int)

    # Auto-add flags with known choices
    choices_map = {
        "scale": ["zscore", "rank", "sign", "madz", "winsor"],
        "max_lookback_data_option": ["common_1200", "specific_long_10k", "full_overlap"],
    }
    added = _add_dataclass_args(p, EvolutionConfig, choices_map=choices_map)
    _add_dataclass_args(p, BacktestConfig, choices_map=choices_map, already_added=added)

    # Pipeline-only flags
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
    # Backtest-only overrides
    p.add_argument("--bt-scale", choices=["zscore", "rank", "sign", "madz", "winsor"],
                   default=None, help="Override backtest scaling (leaves evolution scale unchanged)")

    ns = p.parse_args()
    d = vars(ns)

    evo_kwargs = {k: v for k, v in d.items() if k in EvolutionConfig.__annotations__}
    evo_kwargs["generations"] = ns.generations
    evo_cfg = EvolutionConfig(**evo_kwargs)
    bt_cfg  = BacktestConfig(**{k: v for k, v in d.items()
                                 if k in BacktestConfig.__annotations__})
    # Apply backtest-only overrides
    if getattr(ns, "bt_scale", None):
        bt_cfg.scale = ns.bt_scale

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

    # Optionally generate plots (if matplotlib present)
    try:
        import scripts.diagnostics_plot as diag_plot
        # Prefer explicit run_dir to avoid relying on LATEST
        diag_plot.generate_plots(run_dir)
    except SystemExit:
        pass
    except Exception as e:
        logging.getLogger(__name__).info("Plotting skipped: %s", e)

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
