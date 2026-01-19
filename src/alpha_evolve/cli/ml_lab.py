from __future__ import annotations

import argparse
import json
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Dict

from alpha_evolve.config import BacktestConfig, load_config_file, layer_dataclass_config
from alpha_evolve.config.layering import _flatten_sectioned_config  # type: ignore
from alpha_evolve.ml_lab.runner import parse_run_spec, run_ml_lab
from alpha_evolve.utils.cli import add_dataclass_args


def _load_spec(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("Spec must be a JSON object")
    return payload


def _build_backtest_config(
    spec: Dict[str, Any], cli_overrides: Dict[str, Any]
) -> BacktestConfig:
    config_path = spec.get("config")
    file_cfg: Dict[str, Any] | None = None
    if config_path:
        raw = load_config_file(str(config_path))
        if isinstance(raw, dict) and "backtest" in raw:
            file_cfg = _flatten_sectioned_config(raw, "backtest")
        elif isinstance(raw, dict):
            file_cfg = dict(raw)

    spec_overrides: Dict[str, Any] = {}
    for key in ("backtest_overrides", "backtest"):
        val = spec.get(key)
        if isinstance(val, dict):
            spec_overrides.update(val)
    if "data_dir" in spec and spec["data_dir"]:
        spec_overrides["data_dir"] = spec["data_dir"]

    merged_overrides = dict(spec_overrides)
    merged_overrides.update(cli_overrides)
    layered = layer_dataclass_config(
        BacktestConfig,
        file_cfg=file_cfg,
        env_prefixes=("AE_BT_", "AE_"),
        cli_overrides=merged_overrides,
    )
    return BacktestConfig(**layered)


def _collect_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    fields = {f.name for f in dc_fields(BacktestConfig)}
    for key, value in vars(args).items():
        if key in fields and value is not None:
            overrides[key] = value
    return overrides


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML Lab baselines.")
    parser.add_argument("--spec", required=True, help="Path to ML run spec JSON.")
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for run artefacts."
    )
    parser.add_argument("--config", help="Optional config file override.")
    parser.add_argument("--data_dir", help="Optional data directory override.")
    parser.add_argument("--seed", type=int, help="Override ML seed.")
    parser.add_argument("--train_fraction", type=float, help="Train fraction.")
    parser.add_argument("--train_points", type=int, help="Explicit train points.")
    parser.add_argument("--test_points", type=int, help="Explicit test points.")
    parser.add_argument(
        "--exclude_features",
        help="Comma-separated feature names to exclude from ML.",
    )
    add_dataclass_args(parser, BacktestConfig, skip={"data_dir", "seed"})
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec_path = Path(args.spec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = _load_spec(spec_path)
    if args.config:
        spec["config"] = args.config
    if args.data_dir:
        spec["data_dir"] = args.data_dir
    if args.seed is not None:
        spec["seed"] = int(args.seed)
    if args.train_fraction is not None:
        spec["train_fraction"] = float(args.train_fraction)
    if args.train_points is not None:
        spec["train_points"] = int(args.train_points)
    if args.test_points is not None:
        spec["test_points"] = int(args.test_points)
    if args.exclude_features:
        spec["exclude_features"] = [
            token.strip()
            for token in str(args.exclude_features).split(",")
            if token.strip()
        ]

    spec_out = out_dir / "ml_spec.json"
    spec_out.write_text(json.dumps(spec, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    cli_overrides = _collect_cli_overrides(args)
    cfg = _build_backtest_config(spec, cli_overrides)
    run_spec = parse_run_spec(spec)
    run_ml_lab(cfg, run_spec, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
