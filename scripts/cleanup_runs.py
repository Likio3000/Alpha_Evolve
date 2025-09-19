#!/usr/bin/env python3
"""Utility to prune pipeline run directories and monitor disk usage."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_base_dir(cli_arg: str | None) -> Path:
    env_override = os.environ.get("AE_PIPELINE_DIR") or os.environ.get("AE_OUTPUT_DIR")
    candidate = cli_arg or env_override
    if candidate:
        out = Path(candidate).expanduser()
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()
        else:
            out = out.resolve()
    else:
        out = (PROJECT_ROOT / "pipeline_runs_cs").resolve()
    return out


def _iter_runs(base_dir: Path) -> list[Path]:
    runs = [p for p in base_dir.glob("run_*") if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except FileNotFoundError:
            continue
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune old pipeline runs and report disk usage")
    parser.add_argument("--base-dir", default=None, help="Override pipeline run directory (defaults to AE_PIPELINE_DIR or pipeline_runs_cs)")
    parser.add_argument("--keep", type=int, default=10, help="Number of newest runs to keep regardless of age")
    parser.add_argument("--min-age-days", type=float, default=0.0, help="Only prune runs older than this many days (0 disables age filter)")
    parser.add_argument("--delete", action="store_true", help="Actually delete the selected runs (otherwise dry-run)")
    parser.add_argument("--show-size", action="store_true", help="Compute and display size estimates for runs")

    args = parser.parse_args()

    base_dir = _resolve_base_dir(args.base_dir)
    if not base_dir.exists():
        print(f"Base directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)

    runs = _iter_runs(base_dir)
    if not runs:
        print(f"No runs found under {base_dir}")
        return

    cutoff_ts = None
    if args.min_age_days > 0:
        cutoff_ts = time.time() - args.min_age_days * 86400

    to_prune: list[Path] = []
    for idx, run in enumerate(runs):
        prune = idx >= max(args.keep, 0)
        if cutoff_ts is not None and run.stat().st_mtime < cutoff_ts:
            prune = True
        if prune:
            to_prune.append(run)

    if not to_prune:
        print("Nothing to prune; all runs are within retention policy.")
        return

    action = "Deleting" if args.delete else "Would delete"
    for run in to_prune:
        size_info = ""
        if args.show_size:
            size_info = f" ({_dir_size_bytes(run) / (1024 * 1024):.1f} MiB)"
        print(f"{action}: {run}{size_info}")
        if args.delete:
            shutil.rmtree(run, ignore_errors=True)


if __name__ == "__main__":
    main()
