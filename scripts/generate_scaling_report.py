#!/usr/bin/env python3
"""Build a paper-ready markdown summary from scientific scaling artefacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _fmt(value: Any, ndigits: int = 6) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if v != v:  # NaN
        return "NA"
    return f"{v:.{ndigits}f}"


def _metric_row(result: dict[str, Any]) -> dict[str, Any]:
    ci = result.get("ci95_mean_improvement") or [None, None]
    return {
        "metric": result.get("metric"),
        "n": int(result.get("n", 0) or 0),
        "mean_improvement": result.get("mean_improvement"),
        "ci_lo": ci[0] if len(ci) > 0 else None,
        "ci_hi": ci[1] if len(ci) > 1 else None,
        "p_perm_one_sided": result.get("p_perm_one_sided"),
        "scientific_pass": bool(result.get("scientific_pass", False)),
    }


def _summarize_scientific(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [_metric_row(r) for r in payload.get("results", [])]
    return {
        "path": str(path),
        "control_root": payload.get("control_root"),
        "treatment_root": payload.get("treatment_root"),
        "common_seeds": payload.get("common_seeds", []),
        "rows": rows,
    }


def _summarize_checkpoint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "checkpoint_gens": payload.get("checkpoint_gens", []),
        "pairwise_scientific": payload.get("pairwise_scientific", {}),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate markdown report for scaling experiments")
    p.add_argument(
        "--scientific-json",
        action="append",
        default=[],
        help="Path to scientific_compare JSON (repeatable)",
    )
    p.add_argument(
        "--checkpoint-summary-json",
        action="append",
        default=[],
        help="Path to benchmark checkpoint_summary.json (repeatable)",
    )
    p.add_argument("--title", default="Compute-Scaling Scientific Report")
    p.add_argument("--outdir", default="artifacts/reports")
    p.add_argument("--name", default=None, help="Output basename (without extension)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sci_paths = [Path(p).resolve() for p in args.scientific_json]
    ckpt_paths = [Path(p).resolve() for p in args.checkpoint_summary_json]
    if not sci_paths and not ckpt_paths:
        raise SystemExit("Provide at least one --scientific-json or --checkpoint-summary-json input.")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    basename = args.name or f"scaling_report_{stamp}"
    md_path = outdir / f"{basename}.md"
    json_path = outdir / f"{basename}.json"

    scientific_sections = [_summarize_scientific(p) for p in sci_paths]
    checkpoint_sections = [_summarize_checkpoint(p) for p in ckpt_paths]

    lines: list[str] = [f"# {args.title}", "", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for sec in scientific_sections:
        lines.extend(
            [
                f"## Scientific Compare: `{sec['path']}`",
                "",
                f"- Control: `{sec['control_root']}`",
                f"- Treatment: `{sec['treatment_root']}`",
                f"- Common seeds: {len(sec['common_seeds'])}",
                "",
                "| Metric | n | Mean Improvement | CI95 Low | CI95 High | p(one-sided) | Pass |",
                "|---|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in sec["rows"]:
            lines.append(
                "| {metric} | {n} | {mean} | {lo} | {hi} | {p} | {ps} |".format(
                    metric=row.get("metric", ""),
                    n=int(row.get("n", 0) or 0),
                    mean=_fmt(row.get("mean_improvement")),
                    lo=_fmt(row.get("ci_lo")),
                    hi=_fmt(row.get("ci_hi")),
                    p=_fmt(row.get("p_perm_one_sided"), ndigits=4),
                    ps="yes" if row.get("scientific_pass") else "no",
                )
            )
        lines.append("")

    for sec in checkpoint_sections:
        lines.extend(
            [
                f"## Checkpoint Summary: `{sec['path']}`",
                "",
                f"- Checkpoints: {sec.get('checkpoint_gens', [])}",
                "",
            ]
        )
        pairs = (sec.get("pairwise_scientific") or {}).get("pairs", [])
        if not pairs:
            lines.append("No checkpoint pairwise results found.\n")
            continue
        for pair in pairs:
            lines.extend(
                [
                    f"### Gen {pair.get('from_gen')} -> Gen {pair.get('to_gen')}",
                    "",
                    "| Metric | n | Mean Improvement | CI95 Low | CI95 High | p(one-sided) | Pass |",
                    "|---|---:|---:|---:|---:|---:|:---:|",
                ]
            )
            for row in pair.get("metrics", []):
                ci = row.get("ci95_mean_improvement") or [None, None]
                lines.append(
                    "| {metric} | {n} | {mean} | {lo} | {hi} | {p} | {ps} |".format(
                        metric=row.get("metric", ""),
                        n=int(row.get("n", 0) or 0),
                        mean=_fmt(row.get("mean_improvement")),
                        lo=_fmt(ci[0] if len(ci) > 0 else None),
                        hi=_fmt(ci[1] if len(ci) > 1 else None),
                        p=_fmt(row.get("p_perm_one_sided"), ndigits=4),
                        ps="yes" if row.get("scientific_pass") else "no",
                    )
                )
            lines.append("")

    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "title": args.title,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scientific_sections": scientific_sections,
                "checkpoint_sections": checkpoint_sections,
                "markdown_path": str(md_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[report] Wrote markdown -> {md_path}")
    print(f"[report] Wrote json -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
