#!/usr/bin/env python3
"""
auto_improve.py – simple iterative improvement loop for the crypto pipeline.

What it does
- Empties `pipeline_runs_cs/` (run artefacts) unless `--no-clean` is given.
- Runs `run_pipeline.py` multiple times with small hyperparameter tweaks.
- After each run, parses the backtest summary and keeps the best settings
  based on Sharpe. Iterates for a given number of rounds.

Usage examples
  python scripts/auto_improve.py --iters 2 --gens 5 --data_dir ./data \
    --base_config configs/crypto.toml

  # Quick wiring check without heavy compute
  python scripts/auto_improve.py --iters 1 --gens 1 --dry-run --data_dir tests/data/good \
    --base_config configs/crypto.toml

Notes
- This script optimizes for Sharpe from the backtest summary. It explores a
  few reasonable knobs (selection, novelty, ramping, correlation penalty).
- It also tries opt-in math knobs like structural novelty, per-bar HOF penalty,
  IC t-stat inclusion, recency weighting and rank-softmax schedule.
- It is intentionally conservative to keep turnaround times reasonable.
"""

from __future__ import annotations
import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import itertools
import time
import shutil as _shutil

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline_runs_cs"
_PASSTHROUGH_FLAGS: List[str] = []


@dataclass
class RunResult:
    run_dir: Path
    sharpe_best: float
    params: Dict[str, str]


def _clean_runs_dir() -> None:
    PIPELINE_DIR.mkdir(exist_ok=True)
    for child in PIPELINE_DIR.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            # Best-effort cleanup
            pass


def _read_latest_run_dir() -> Optional[Path]:
    latest_file = PIPELINE_DIR / "LATEST"
    try:
        if latest_file.exists():
            p = latest_file.read_text().strip()
            if p:
                run_path = Path(p)
                return run_path if run_path.exists() else None
    except Exception:
        pass
    return None


def _read_best_sharpe_from_run(run_dir: Path) -> Optional[float]:
    bt_dir = run_dir / "backtest_portfolio_csvs"
    # Find any backtest summary CSV; prefer the one matching the largest topN
    candidates = sorted(bt_dir.glob("backtest_summary_top*.csv"))
    if not candidates:
        return None
    # Take the last (alphabetically) which usually corresponds to highest N
    csv_path = candidates[-1]
    try:
        with open(csv_path, newline="") as fh:
            rdr = csv.DictReader(fh)
            best = None
            for row in rdr:
                try:
                    s = float(row.get("Sharpe", "nan"))
                except Exception:
                    continue
                if best is None or s > best:
                    best = s
            return best
    except Exception:
        return None


def _build_cli_args(
    gens: int,
    base_config: Optional[str],
    data_dir: Optional[str],
    bt_top: int,
    extra: Dict[str, str],
    dry_run: bool,
    passthrough: Optional[List[str]] = None,
) -> List[str]:
    # Choose runner: prefer uv if available to ensure deps are resolved
    uv_path = _shutil.which("uv")
    # Allow forcing system Python even if uv is present (useful when a local
    # virtualenv is already set up): export AUTO_IMPROVE_FORCE_PY=1
    force_py = os.environ.get("AUTO_IMPROVE_FORCE_PY", "0") in ("1", "true", "True")
    if uv_path and not force_py:
        args: List[str] = [uv_path, "run", "run_pipeline.py", str(gens)]
    else:
        args = [sys.executable, str(ROOT / "run_pipeline.py"), str(gens)]
    if base_config:
        args += ["--config", base_config]
    if data_dir:
        # Applies to both evolution and backtest via config layering
        args += ["--data_dir", data_dir]
    # Backtest override
    args += ["--top_to_backtest", str(bt_top)]
    # Optional dry-run to validate wiring without compute
    if dry_run:
        args.append("--dry-run")
    # Add all extra key/value overrides (cli overrides > env > file)
    for k, v in extra.items():
        flag = f"--{k}"
        # Accept native booleans and string-y booleans
        if isinstance(v, str) and v.lower() in ("true", "false"):
            v = (v.lower() == "true")
        if isinstance(v, bool):
            if v:
                args.append(flag)
            else:
                # Support convenient negation if defaults to True in dataclass
                args.append(f"--no-{k}")
        else:
            args += [flag, str(v)]
    # Append any passthrough flags at the end (highest precedence)
    if passthrough:
        args += list(passthrough)
    return args


def _run_pipeline_once(
    gens: int,
    base_config: Optional[str],
    data_dir: Optional[str],
    bt_top: int,
    extra: Dict[str, str],
    dry_run: bool,
) -> Tuple[Optional[Path], Optional[float]]:
    # passthrough flags are read in main() and bound via a closure
    global _PASSTHROUGH_FLAGS
    args = _build_cli_args(gens, base_config, data_dir, bt_top, extra, dry_run, _PASSTHROUGH_FLAGS)
    env = os.environ.copy()
    # Ensure caching works consistently across runs unless user overrides
    env.setdefault("AE_DISABLE_ALIGN_CACHE", "0")
    try:
        subprocess.run(args, cwd=str(ROOT), check=True)
    except subprocess.CalledProcessError:
        return None, None
    run_dir = _read_latest_run_dir()
    if not run_dir or dry_run:
        return run_dir, None
    sharpe = _read_best_sharpe_from_run(run_dir)
    return run_dir, sharpe


ParamVal = Union[str, bool, int, float]


def _candidate_perturbations(base: Dict[str, ParamVal]) -> List[Dict[str, ParamVal]]:
    """Return a small list of candidate tweaks around the current base params."""
    b = {**base}
    # Coerce some defaults if not present
    b.setdefault("selection_metric", "auto")
    b.setdefault("ramp_fraction", "0.33")
    b.setdefault("ramp_min_gens", "5")
    b.setdefault("novelty_boost_w", "0.02")
    b.setdefault("novelty_struct_w", "0.00")
    b.setdefault("hof_corr_mode", "flat")
    b.setdefault("ic_tstat_w", "0.00")
    b.setdefault("temporal_decay_half_life", "0")
    b.setdefault("rank_softmax_beta_floor", "0.0")
    b.setdefault("rank_softmax_beta_target", "2.0")
    b.setdefault("corr_penalty_w", "0.35")
    b.setdefault("fresh_rate", "0.12")
    b.setdefault("pop_size", "100")
    b.setdefault("hof_per_gen", "3")
    # New math defaults to improve search and robustness
    b.setdefault("moea_enabled", True)
    b.setdefault("moea_elite_frac", "0.25")
    b.setdefault("mf_enabled", True)
    b.setdefault("mf_initial_fraction", "0.4")
    b.setdefault("mf_promote_fraction", "0.3")
    b.setdefault("mf_min_promote", "8")
    b.setdefault("cv_k_folds", "4")
    b.setdefault("cv_embargo", "5")

    cands: List[Dict[str, str]] = []
    # 0) current base as-is
    cands.append({**b})
    # 1) more exploration via novelty boost
    c = {**b}
    c["novelty_boost_w"] = "0.05"
    cands.append(c)
    # 2) gentler ramp (longer exploration)
    c = {**b}
    c["ramp_fraction"] = "0.5"
    cands.append(c)
    # 3) a tad lower corr penalty to encourage diversity
    c = {**b}
    c["corr_penalty_w"] = "0.25"
    cands.append(c)
    # 4) phased selection with an IC-only warmup
    c = {**b}
    c["selection_metric"] = "phased"
    c["ic_phase_gens"] = "5"
    cands.append(c)
    # 5) increase fresh_rate (more new programs)
    c = {**b}
    c["fresh_rate"] = "0.20"
    cands.append(c)
    # 6) larger population size
    c = {**b}
    c["pop_size"] = "150"
    cands.append(c)
    # 7) more HOF items per generation
    c = {**b}
    c["hof_per_gen"] = "5"
    cands.append(c)
    # 8) combined capacity bump
    c = {**b}
    c.update({"fresh_rate": "0.20", "pop_size": "150", "hof_per_gen": "5"})
    cands.append(c)
    # 9) structural novelty bonus
    c = {**b}
    c["novelty_struct_w"] = "0.02"
    cands.append(c)
    # 10) per-bar HOF correlation penalty mode
    c = {**b}
    c["hof_corr_mode"] = "per_bar"
    cands.append(c)
    # 11) include IC t-stat in fitness
    c = {**b}
    c["ic_tstat_w"] = "0.5"
    cands.append(c)
    # 12) add recency weighting (half-life)
    c = {**b}
    c["temporal_decay_half_life"] = "100"
    cands.append(c)
    # 13) stronger late-stage selection pressure
    c = {**b}
    c["rank_softmax_beta_target"] = "3.0"
    cands.append(c)
    # 14) enable MOEA elites
    c = {**b}
    c.update({"moea_enabled": True, "moea_elite_frac": "0.25"})
    cands.append(c)
    # 15) enable multi-fidelity
    c = {**b}
    c.update({"mf_enabled": True, "mf_initial_fraction": "0.4", "mf_promote_fraction": "0.3"})
    cands.append(c)
    # 16) enable purged CV
    c = {**b}
    c.update({"cv_k_folds": "4", "cv_embargo": "5"})
    cands.append(c)
    return cands


def main() -> None:
    ap = argparse.ArgumentParser(description="Iterative improvement loop for crypto pipeline", add_help=True)
    ap.add_argument("--iters", type=int, default=2, help="Number of improvement rounds")
    ap.add_argument("--gens", type=int, default=10, help="Generations per pipeline run")
    ap.add_argument("--base_config", type=str, default=None, help="TOML/YAML base config")
    ap.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    ap.add_argument("--bt_top", type=int, default=10, help="Top N alphas to backtest")
    ap.add_argument("--no-clean", action="store_true", help="Do not delete existing runs in pipeline_runs_cs")
    ap.add_argument("--dry-run", action="store_true", help="Use pipeline dry-run to validate wiring only")
    # Sweep mode for capacity knobs
    ap.add_argument("--sweep-capacity", action="store_true", help="Grid-sweep fresh_rate, pop_size, hof_per_gen")
    ap.add_argument("--seeds", type=str, default=None, help="Optional comma-separated seeds for sweep (e.g., '7,42,99')")
    ap.add_argument("--out-summary", type=str, default=None, help="Optional path to write a CSV summary of sweeps")
    # Accept and forward any additional run_pipeline flags transparently
    args, passthrough = ap.parse_known_args()
    # Store passthrough flags globally so helper can access
    global _PASSTHROUGH_FLAGS
    _PASSTHROUGH_FLAGS = [x for x in passthrough if x != "--"]

    if not args.no_clean:
        print("Cleaning pipeline_runs_cs/ …")
        _clean_runs_dir()

    # Start with a reasonable crypto-leaning base
    base_params: Dict[str, ParamVal] = {
        "selection_metric": "auto",
        "ramp_fraction": "0.33",
        "ramp_min_gens": "5",
        "novelty_boost_w": "0.02",
        # Crypto friendly defaults
        "sector_neutralize": True,
        # Keep evaluation scalable early on
        "workers": "1",
        # Capacity defaults
        "fresh_rate": "0.12",
        "pop_size": "100",
        "hof_per_gen": "3",
    }

    best_overall: Optional[RunResult] = None

    # Optional sweep over capacity knobs (fresh_rate, pop_size, hof_per_gen)
    if args.sweep_capacity:
        seeds: List[int] = []
        if args.seeds:
            try:
                seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
            except Exception:
                seeds = []
        if not seeds:
            seeds = [base_params.get("seed", 42)] if isinstance(base_params.get("seed", 42), int) else [42]

        fr_vals = [str(base_params.get("fresh_rate", "0.12")), "0.20"]
        ps_vals = [str(base_params.get("pop_size", "100")), "150"]
        hof_vals = [str(base_params.get("hof_per_gen", "3")), "5"]

        rows: List[Dict[str, str]] = []
        grid = list(itertools.product(fr_vals, ps_vals, hof_vals, seeds))
        print(f"Capacity sweep: {len(grid)} combos × gens={args.gens}")

        for (fr, ps, hof, seed) in grid:
            cand = {**base_params, "fresh_rate": fr, "pop_size": ps, "hof_per_gen": hof, "seed": str(seed)}
            print(f"→ Sweep cand seed={seed} fr={fr} ps={ps} hof={hof}")
            run_dir, sharpe = _run_pipeline_once(
                gens=args.gens,
                base_config=args.base_config,
                data_dir=args.data_dir,
                bt_top=args.bt_top,
                extra=cand,
                dry_run=args.dry_run,
            )
            row = {
                "seed": str(seed),
                "fresh_rate": str(fr),
                "pop_size": str(ps),
                "hof_per_gen": str(hof),
                "sharpe_best": "" if sharpe is None else f"{sharpe:+.6f}",
                "run_dir": "" if run_dir is None else str(run_dir),
            }
            rows.append(row)

        # Write summary CSV
        out_path = args.out_summary
        if not out_path:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_dir = PIPELINE_DIR / "sweeps"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(out_dir / f"capacity_sweep_{stamp}.csv")
        try:
            with open(out_path, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=["seed", "fresh_rate", "pop_size", "hof_per_gen", "sharpe_best", "run_dir"])
                wr.writeheader()
                wr.writerows(rows)
            print(f"Saved sweep summary → {out_path}")
        except Exception as e:
            print(f"Failed to save sweep summary: {e}")
        return

    for it in range(1, args.iters + 1):
        print(f"\n=== Iteration {it}/{args.iters} ===")
        candidates = _candidate_perturbations(base_params)
        round_best: Optional[RunResult] = None

        for idx, cand in enumerate(candidates, 1):
            print(f"→ Candidate {idx}/{len(candidates)}: {cand}")
            # Build args and append passthrough flags so users can set any
            # additional knobs supported by run_pipeline/backtest.
            run_dir, sharpe = _run_pipeline_once(
                gens=args.gens,
                base_config=args.base_config,
                data_dir=args.data_dir,
                bt_top=args.bt_top,
                extra=cand,
                dry_run=args.dry_run,
            )
            # No second attempt needed; passthrough already included
            # On dry-run we cannot score; just record run_dir
            if args.dry_run:
                print(f"   completed (dry-run). run_dir={run_dir}")
                continue
            if run_dir is None or sharpe is None:
                print("   failed or missing summary; skipping.")
                continue
            print(f"   Sharpe(best) = {sharpe:+.3f}  run_dir={run_dir}")
            if round_best is None or sharpe > round_best.sharpe_best:
                round_best = RunResult(run_dir=run_dir, sharpe_best=sharpe, params=cand)

        if args.dry_run:
            # In dry-run we don't update params; just continue
            continue

        if round_best is None:
            print("No successful candidates this round; keeping previous params.")
            continue

        # Adopt best params for next iteration
        base_params = {**round_best.params}
        if best_overall is None or round_best.sharpe_best > best_overall.sharpe_best:
            best_overall = round_best
        print(f"✓ Round best Sharpe {round_best.sharpe_best:+.3f}; params → {base_params}")

    if not args.dry_run and best_overall is not None:
        print("\n=== Best overall ===")
        print(f"Sharpe(best) {best_overall.sharpe_best:+.3f}")
        print(f"Run dir: {best_overall.run_dir}")
        print(f"Params: {best_overall.params}")


if __name__ == "__main__":
    main()
