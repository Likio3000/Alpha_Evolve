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
import os
import pickle
import sys
import time
from pathlib import Path
from dataclasses import dataclass, fields as dc_fields

from config import EvolutionConfig, BacktestConfig
from utils.data_loading_common import DataLoadError
import evolve_alphas as ae
import backtest_evolved_alphas as bt
from utils.logging_setup import setup_logging
import json as _json
from utils.cli import add_dataclass_args
from utils.config_layering import load_config_file, layer_dataclass_config, _flatten_sectioned_config  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pipeline_runs_cs"


@dataclass
class PipelineOptions:
    """Runtime controls for programmatic pipeline execution."""

    debug_prints: bool = False
    run_baselines: bool = False
    retrain_baselines: bool = False
    log_level: str = "INFO"
    log_file: str | None = None
    dry_run: bool = False
    output_dir: str | None = None
    persist_hof_per_gen: bool = True
    disable_align_cache: bool = False
    align_cache_dir: str | None = None


def _resolve_output_dir(cli_arg: str | None) -> Path:
    """Resolve the pipeline output directory from CLI or env overrides."""
    env_override = os.environ.get("AE_PIPELINE_DIR") or os.environ.get("AE_OUTPUT_DIR")
    candidate = cli_arg or env_override
    if candidate:
        out = Path(candidate).expanduser()
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()
        else:
            out = out.resolve()
    else:
        out = DEFAULT_OUTPUT_DIR.resolve()
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  CLI → two dataclass configs (auto-generated from dataclasses)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> tuple[EvolutionConfig, BacktestConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description="Evolve and back-test alphas (one-stop shop)")

    # Positional generations
    p.add_argument("generations", type=int)

    # Auto-add flags with known choices
    choices_map = {
        "scale": ["zscore", "rank", "sign", "madz", "winsor"],
        "max_lookback_data_option": ["common_1200", "specific_long_10k", "full_overlap"],
    }
    added = add_dataclass_args(p, EvolutionConfig, choices_map=choices_map)
    add_dataclass_args(p, BacktestConfig, choices_map=choices_map, already_added=added)

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
    p.add_argument("--print-config", action="store_true",
                   help="Print resolved Evolution and Backtest configs as JSON and exit")
    p.add_argument("--output-dir", default=None,
                   help="Base directory for pipeline run artefacts (defaults to pipeline_runs_cs relative to the project root)."
                   )
    # Cache controls
    p.add_argument("--disable-align-cache", action="store_true",
                   help="Disable alignment cache (equivalent to AE_DISABLE_ALIGN_CACHE=1)")
    p.add_argument("--align-cache-dir", default=None,
                   help="Custom directory for alignment cache (AE_ALIGN_CACHE_DIR)")
    # Backtest-only overrides
    p.add_argument("--bt-scale", choices=["zscore", "rank", "sign", "madz", "winsor"],
                   default=None, help="Override backtest scaling (leaves evolution scale unchanged)")
    # HOF snapshot persistence toggle
    p.add_argument("--persist-hof-per-gen", dest="persist_hof_per_gen", action="store_true", default=True,
                   help="Persist per-generation HOF snapshots under run_dir/meta")
    p.add_argument("--no-persist-hof-per-gen", dest="persist_hof_per_gen", action="store_false")

    p.add_argument("--config", default=None,
                   help="Optional TOML/YAML config file (file < env < CLI)")

    ns = p.parse_args()
    d = vars(ns)

    # Collect full field-name sets including inherited dataclass fields.
    evo_field_names = {f.name for f in dc_fields(EvolutionConfig)}
    bt_field_names = {f.name for f in dc_fields(BacktestConfig)}

    cli_evo = {k: v for k, v in d.items() if k in evo_field_names}
    cli_bt  = {k: v for k, v in d.items() if k in bt_field_names}
    cli_evo["generations"] = ns.generations

    # Load config file if provided and split sections
    evo_file_cfg: dict | None = None
    bt_file_cfg: dict | None = None
    if getattr(ns, "config", None):
        raw = load_config_file(ns.config)
        evo_file_cfg = _flatten_sectioned_config(raw, "evolution") if "evolution" in raw else None
        bt_file_cfg = _flatten_sectioned_config(raw, "backtest") if "backtest" in raw else None
        # If no explicit sections, use flat config for both as baseline
        if evo_file_cfg is None and bt_file_cfg is None:
            flat = _flatten_sectioned_config(raw, None)
            evo_file_cfg = flat
            bt_file_cfg = flat

    evo_kwargs = layer_dataclass_config(
        EvolutionConfig,
        file_cfg=evo_file_cfg,
        env_prefixes=("AE_", "AE_EVO_"),
        cli_overrides=cli_evo,
    )
    bt_kwargs = layer_dataclass_config(
        BacktestConfig,
        file_cfg=bt_file_cfg,
        env_prefixes=("AE_", "AE_BT_"),
        cli_overrides=cli_bt,
    )

    evo_cfg = EvolutionConfig(**evo_kwargs)
    bt_cfg  = BacktestConfig(**bt_kwargs)
    # Apply backtest-only overrides
    if getattr(ns, "bt_scale", None):
        bt_cfg.scale = ns.bt_scale

    return evo_cfg, bt_cfg, ns


# ─────────────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────────────
def _write_summary_json(run_dir: Path, pickle_path: Path, summary_csv: Path) -> Path:
    """Write a compact run summary JSON with key artefacts.

    Returns path to the created SUMMARY.json.
    """
    summary_json = summary_csv.with_suffix(".json")
    data = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "programs_pickle": str(pickle_path),
        "backtest_summary_csv": str(summary_csv),
        "backtest_summary_json": str(summary_json),
    }
    meta_dir = run_dir / "meta"
    if (meta_dir / "data_alignment.json").exists():
        data["data_alignment"] = str(meta_dir / "data_alignment.json")
    # Optionally include counts
    try:
        import pandas as _pd
        df = _pd.read_csv(summary_csv)
        data["backtested_alphas"] = int(len(df))
        # Best Sharpe and a few handy fields for UI consumption
        if "Sharpe" in df.columns and len(df) > 0:
            best = df.sort_values("Sharpe", ascending=False).iloc[0]
            data["best_metrics"] = {
                "Sharpe": float(best.get("Sharpe", 0.0)),
                "AnnReturn": float(best.get("AnnReturn", 0.0)),
                "MaxDD": float(best.get("MaxDD", 0.0)),
                "Ops": int(best.get("Ops", 0)),
                "AlphaID": str(best.get("AlphaID", "")),
                "TimeseriesFile": str(best.get("TimeseriesFile", "")),
            }
    except Exception:
        pass
    out = run_dir / "SUMMARY.json"
    with open(out, "w") as fh:
        _json.dump(data, fh, indent=2)
    return out


def _write_hof_snapshots(run_dir: Path) -> list[Path]:
    """Write per-generation HOF snapshots into run_dir/meta/.

    Uses utils.diagnostics.get_all() entries produced during evolution. Each
    entry may include a 'hof' field added by evolve_alphas; if present, this
    function writes meta/hof_gen_<N>.json where N is generation.
    Returns the list of written file paths.
    """
    from utils import diagnostics as diag
    out_paths: list[Path] = []
    try:
        entries = diag.get_all()
    except Exception:
        entries = []
    if not entries:
        return out_paths
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    for ent in entries:
        try:
            gen = int(ent.get("generation", 0))
            hof = ent.get("hof")
            if not hof or not gen:
                continue
            p = meta_dir / f"hof_gen_{gen:03d}.json"
            with open(p, "w") as fh:
                _json.dump(hof, fh, indent=2)
            out_paths.append(p)
        except Exception:
            continue
    return out_paths
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


def _write_data_alignment_meta(run_dir: Path, evo_cfg: EvolutionConfig) -> Path:
    """Persist standardized data alignment diagnostics.

    Always writes a JSON file with config-level provenance; if diagnostics are
    available from the evolution loader they are included.
    """
    from evolution_components import data_handling as evo_dh
    di = None
    try:
        di = evo_dh.get_data_diagnostics()
    except Exception:
        di = None

    payload = {
        "data_dir": evo_cfg.data_dir,
        "strategy": evo_cfg.max_lookback_data_option,
        "min_common_points": evo_cfg.min_common_points,
        "eval_lag": evo_cfg.eval_lag,
        "source": "evolution_loader",
    }
    if di is not None:
        payload.update({
            "n_symbols_before": di.n_symbols_before,
            "n_symbols_after": di.n_symbols_after,
            "dropped_symbols": di.dropped_symbols,
            "overlap_len": di.overlap_len,
            "overlap_start": str(di.overlap_start) if di.overlap_start is not None else None,
            "overlap_end": str(di.overlap_end) if di.overlap_end is not None else None,
        })
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    out = meta_dir / "data_alignment.json"
    with open(out, "w") as fh:
        _json.dump(payload, fh, indent=2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  programmatic API
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline_programmatic(
    evo_cfg: EvolutionConfig,
    bt_cfg: BacktestConfig,
    options: PipelineOptions | None = None,
) -> Path:
    opts = options or PipelineOptions()

    if opts.disable_align_cache:
        os.environ["AE_DISABLE_ALIGN_CACHE"] = "1"
    if opts.align_cache_dir:
        os.environ["AE_ALIGN_CACHE_DIR"] = str(opts.align_cache_dir)

    level = getattr(logging, str(opts.log_level).upper(), logging.INFO)
    if opts.debug_prints:
        level = logging.DEBUG
    elif getattr(evo_cfg, "quiet", False):
        level = logging.WARNING

    setup_logging(level=level, log_file=opts.log_file)
    logger = logging.getLogger(__name__)

    base_output_dir = _resolve_output_dir(opts.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AE_PIPELINE_DIR"] = str(base_output_dir)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (
        base_output_dir
        / f"run_g{evo_cfg.generations}_seed{evo_cfg.seed}_"
        f"{evo_cfg.max_lookback_data_option}_{run_stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        import json
        import platform
        import subprocess
        from dataclasses import asdict

        (run_dir / "meta").mkdir(exist_ok=True)
        with open(run_dir / "meta" / "evolution_config.json", "w") as fh:
            json.dump(asdict(evo_cfg), fh, indent=2)
        with open(run_dir / "meta" / "backtest_config.json", "w") as fh:
            json.dump(asdict(bt_cfg), fh, indent=2)

        meta: dict[str, object] = {
            "python": sys.version,
            "platform": platform.platform(),
            "time": run_stamp,
        }
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            meta["git_commit"] = sha
        except Exception:
            pass
        try:
            import importlib.metadata as ilmd

            pkgs = []
            for dist in sorted(
                ilmd.distributions(),
                key=lambda d: d.metadata["Name"].lower() if d.metadata and d.metadata.get("Name") else "",
            ):
                try:
                    pkgs.append({"name": dist.metadata["Name"], "version": dist.version})
                except Exception:
                    continue
            if pkgs:
                meta["packages"] = pkgs
        except Exception:
            pass
        with open(run_dir / "meta" / "run_metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        try:
            latest_path = base_output_dir / "LATEST"
            value = str(run_dir.resolve())
            try:
                rel_to_root = run_dir.resolve().relative_to(PROJECT_ROOT)
                value = str(rel_to_root)
            except ValueError:
                value = str(run_dir.resolve())
            with open(latest_path, "w") as fh:
                fh.write(value)
        except Exception:
            logger.warning("Failed to update LATEST pointer", exc_info=True)
    except Exception:
        logger.warning("Failed to save run metadata.")

    if opts.dry_run:
        logger.info(
            "Dry run: would evolve %d generations and backtest top %d; outputs → %s",
            evo_cfg.generations,
            bt_cfg.top_to_backtest,
            run_dir,
        )
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
        return run_dir

    try:
        pickle_path = _evolve_and_save(evo_cfg, run_dir)
    except DataLoadError as e:
        logger.exception("Evolution failed due to data loading error: %s", e)
        raise
    except Exception as e:
        logger.exception("Evolution failed: %s", e)
        raise

    try:
        _write_data_alignment_meta(run_dir, evo_cfg)
    except Exception:
        logger.warning("Failed to write data alignment metadata.")

    try:
        from utils import diagnostics as diag

        diags = diag.get_all()
        if diags:
            import json

            diag_path = run_dir / "diagnostics.json"
            with open(diag_path, "w") as fh:
                json.dump(diags, fh, indent=2)
            logger.info("Saved diagnostics → %s", diag_path)
            if opts.persist_hof_per_gen:
                try:
                    hof_paths = _write_hof_snapshots(run_dir)
                    if hof_paths:
                        logger.info("Saved %d HOF snapshots under meta/", len(hof_paths))
                except Exception:
                    pass
    except Exception:
        pass

    try:
        import scripts.diagnostics_plot as diag_plot

        diag_plot.generate_plots(run_dir)
    except SystemExit:
        pass
    except Exception as e:
        logger.info("Plotting skipped: %s", e)

    logger.info("\n— Back-testing …")
    try:
        summary_csv_path = bt.run(
            bt_cfg,
            outdir=run_dir / "backtest_portfolio_csvs",
            programs_pickle=pickle_path,
            debug_prints=opts.debug_prints,
            annualization_factor_override=None,
            logger=logger,
        )
    except Exception as e:
        logger.exception("Back-testing failed: %s", e)
        raise

    try:
        summary_path = _write_summary_json(run_dir, pickle_path, summary_csv_path)
        logger.info("Wrote run summary → %s", summary_path)
    except Exception:
        logger.info("Failed to write SUMMARY.json; continuing.")

    try:
        from scripts.backtest_diagnostics_plot import plot_alpha_timeseries

        bt_dir = run_dir / "backtest_portfolio_csvs"
        csvs = sorted(bt_dir.glob("alpha_*_timeseries.csv"))
        for c in csvs:
            out_png = plot_alpha_timeseries(c)
            logger.info("Saved plot → %s", out_png)
    except SystemExit:
        pass
    except Exception as e:
        logger.info("Backtest plots skipped: %s", e)

    if opts.run_baselines:
        _train_baselines(
            bt_cfg.data_dir,
            run_dir,
            retrain=opts.retrain_baselines,
        )

    logger.info("\n✔  Pipeline finished – artefacts in  %s", run_dir)
    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    evo_cfg, bt_cfg, cli = parse_args()

    # Print merged configuration and exit early (no logging, no side-effects)
    if getattr(cli, "print_config", False):
        from dataclasses import asdict
        payload = {"evolution": asdict(evo_cfg), "backtest": asdict(bt_cfg)}
        print(_json.dumps(payload, indent=2))
        return
    options = PipelineOptions(
        debug_prints=getattr(cli, "debug_prints", False),
        run_baselines=getattr(cli, "run_baselines", False),
        retrain_baselines=getattr(cli, "retrain_baselines", False),
        log_level=getattr(cli, "log_level", "INFO"),
        log_file=getattr(cli, "log_file", None),
        dry_run=getattr(cli, "dry_run", False),
        output_dir=getattr(cli, "output_dir", None),
        persist_hof_per_gen=getattr(cli, "persist_hof_per_gen", True),
        disable_align_cache=getattr(cli, "disable_align_cache", False),
        align_cache_dir=getattr(cli, "align_cache_dir", None),
    )

    try:
        run_pipeline_programmatic(evo_cfg, bt_cfg, options)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
