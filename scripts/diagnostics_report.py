#!/usr/bin/env python
"""
diagnostics_report.py – Summarize an evolution run's diagnostics.json.

Usage:
  uv run scripts/diagnostics_report.py /path/to/run_dir

If no path is given, reads the path from pipeline_runs_cs/LATEST.
Outputs a concise text summary to stdout and writes a CSV with per-generation
metrics next to diagnostics.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import csv


def _resolve_run_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if p.is_dir():
            return p
        raise SystemExit(f"Run dir does not exist: {p}")
    latest = Path("pipeline_runs_cs") / "LATEST"
    if latest.exists():
        return Path(latest.read_text().strip())
    raise SystemExit("No run dir argument and LATEST not found.")


def main() -> None:
    run_dir = _resolve_run_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    diag_path = run_dir / "diagnostics.json"
    if not diag_path.exists():
        raise SystemExit(f"diagnostics.json not found in {run_dir}")

    with open(diag_path) as fh:
        diags = json.load(fh)

    rows = []
    for d in diags:
        g = int(d.get("generation", -1))
        best = d.get("best", {}) or {}
        q = d.get("pop_quantiles", {}) or {}
        ramp = d.get("ramp", {}) or {}
        rows.append({
            "generation": g,
            "best_fitness": best.get("fitness"),
            "best_ic": best.get("mean_ic"),
            "best_ops": best.get("ops"),
            "best_ic_std": best.get("ic_std"),
            "best_turnover": best.get("turnover"),
            "best_parsimony": best.get("parsimony"),
            "best_corr_pen": best.get("corr_pen"),
            "q_best": q.get("best"),
            "q_p95": q.get("p95"),
            "q_p75": q.get("p75"),
            "q_median": q.get("median"),
            "q_p25": q.get("p25"),
            "q_count": q.get("count"),
            "r_corr_w": ramp.get("corr_w"),
            "r_ic_std_w": ramp.get("ic_std_w"),
            "r_turnover_w": ramp.get("turnover_w"),
            "r_sharpe_w": ramp.get("sharpe_w"),
            "eval_seconds": d.get("gen_eval_seconds"),
        })

    out_csv = run_dir / "diagnostics_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Console summary
    print(f"Run: {run_dir}")
    if rows:
        print(f"Generations: {rows[0]['generation']}..{rows[-1]['generation']}")
        last = rows[-1]
        print(f"Last gen best fitness: {last['best_fitness']:.4f} IC: {last['best_ic']:.4f} ops: {last['best_ops']}")
        print(f"Median fitness trajectory (first→last): {rows[0]['q_median']:.4f} → {last['q_median']:.4f}")
    print(f"Saved CSV → {out_csv}")


if __name__ == "__main__":
    main()

