#!/usr/bin/env python3
"""Self-play coordinator for automated parameter tuning runs.

This script orchestrates a series of pipeline runs with sampled overrides,
records the resulting metrics, and appends them to
`pipeline_runs_cs/selfplay_history.json` for the dashboard to consume.

Example usage:
    uv run scripts/selfplay.py \
        --dataset crypto \
        --iterations 3 \
        --search-file configs/selfplay_search.json

The search file should be JSON with the following shape:
{
  "base_overrides": {"generations": 5, "pop_size": 120},
  "search_space": {
      "fresh_rate": [0.1, 0.2, 0.25],
      "p_mut": {"min": 0.7, "max": 0.95, "step": 0.05},
      "factor_penalty_ic": [0.0, 0.01, 0.02]
  },
  "metric": "Sharpe"
}
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from scripts.dashboard_server.helpers import (
    ROOT,
    PIPELINE_DIR,
    build_pipeline_args,
    resolve_latest_run_dir,
)

HISTORY_PATH = PIPELINE_DIR / "selfplay_history.json"


@dataclass
class SearchSpace:
    base_overrides: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, Any] = field(default_factory=dict)
    metric: str = "Sharpe"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SearchSpace":
        base = dict(data.get("base_overrides") or {})
        space = dict(data.get("search_space") or {})
        metric = str(data.get("metric") or "Sharpe")
        return cls(base_overrides=base, search_space=space, metric=metric)


class SelfPlayHistory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    self.entries = loaded
            except Exception:  # pragma: no cover - best effort read
                self.entries = []

    def append(self, record: Dict[str, Any]) -> None:
        self.entries.append(record)
        self.entries.sort(key=lambda item: (item.get("timestamp"), item.get("iteration", -1)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")


class SelfPlayCoordinator:
    def __init__(
        self,
        dataset: str,
        iterations: int,
        search: SearchSpace,
        config_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.iterations = iterations
        self.search = search
        self.config_path = config_path
        self.data_dir = data_dir
        if seed is not None:
            random.seed(seed)
        self.history = SelfPlayHistory(HISTORY_PATH)

    def run(self) -> None:
        existing = {p.resolve() for p in PIPELINE_DIR.glob("run_*")}
        for iteration in range(self.iterations):
            overrides = self._sample_overrides()
            payload = self._build_payload(overrides)
            args = build_pipeline_args(payload)
            print(f"[selfplay] Iteration {iteration + 1}/{self.iterations} – overrides={overrides}")
            result = subprocess.run(args, cwd=str(ROOT))
            if result.returncode != 0:
                raise RuntimeError(f"Pipeline run failed with exit code {result.returncode}")
            run_dir = self._resolve_new_run(existing)
            summary = self._load_summary(run_dir)
            metrics = summary.get("best_metrics", {}) if summary else {}
            record = {
                "iteration": len(self.history.entries),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "dataset": self.dataset,
                "overrides": overrides,
                "metric": self.search.metric,
                "metrics": metrics,
                "run_dir": self._format_run_dir(run_dir),
                "backtested_alphas": summary.get("backtested_alphas") if summary else None,
                "pipeline_args": args,
            }
            self.history.append(record)
            existing.add(run_dir.resolve())
            print(
                f"[selfplay] Iteration {iteration + 1} completed – "
                f"Sharpe={metrics.get('Sharpe')}, run_dir={record['run_dir']}"
            )

    def _sample_overrides(self) -> Dict[str, Any]:
        overrides = dict(self.search.base_overrides)
        for key, spec in self.search.search_space.items():
            if isinstance(spec, Mapping):
                low = float(spec.get("min", spec.get("low", 0.0)))
                high = float(spec.get("max", spec.get("high", low)))
                step = float(spec.get("step", 0.0))
                if step:
                    count = max(1, int(round((high - low) / step)))
                    choices = [low + step * i for i in range(count + 1)]
                    overrides[key] = random.choice(choices)
                else:
                    overrides[key] = random.uniform(low, high)
            elif isinstance(spec, Iterable) and not isinstance(spec, (str, bytes)):
                candidates = list(spec)
                if not candidates:
                    continue
                overrides[key] = random.choice(candidates)
            else:
                overrides[key] = spec
        return overrides

    def _build_payload(self, overrides: MutableMapping[str, Any]) -> Dict[str, Any]:
        overrides_copy = dict(overrides)
        payload: Dict[str, Any] = {
            "dataset": self.dataset,
            "overrides": overrides_copy,
        }
        generations = overrides_copy.get("generations", self.search.base_overrides.get("generations", 5))
        payload["generations"] = generations
        if self.config_path:
            payload["config"] = self.config_path
        if self.data_dir:
            payload["data_dir"] = self.data_dir
        return payload

    def _resolve_new_run(self, existing: set[Path]) -> Path:
        latest = resolve_latest_run_dir()
        candidates = {p.resolve() for p in PIPELINE_DIR.glob("run_*")}
        new_dirs = [p for p in candidates if p not in existing]
        if new_dirs:
            run_path = sorted(new_dirs)[-1]
        elif latest is not None:
            run_path = latest.resolve()
        else:
            raise RuntimeError("Unable to locate newly created run directory")
        return run_path

    def _load_summary(self, run_dir: Path) -> Dict[str, Any]:
        summary_path = run_dir / "SUMMARY.json"
        if not summary_path.exists():
            return {}
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _format_run_dir(self, run_dir: Path) -> str:
        try:
            return str(run_dir.resolve().relative_to(ROOT))
        except ValueError:
            return str(run_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play parameter tuner for run_pipeline")
    parser.add_argument("--dataset", required=True, help="Preset dataset key (e.g., crypto, sp500)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations to launch")
    parser.add_argument("--search-file", type=str, required=True, help="Path to JSON search specification")
    parser.add_argument("--config", type=str, default=None, help="Optional explicit config path")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional data directory override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_path = Path(args.search_file)
    if not search_path.exists():
        raise SystemExit(f"Search spec not found: {search_path}")
    search_spec = json.loads(search_path.read_text(encoding="utf-8"))
    search = SearchSpace.from_mapping(search_spec)
    coordinator = SelfPlayCoordinator(
        dataset=args.dataset,
        iterations=args.iterations,
        search=search,
        config_path=args.config,
        data_dir=args.data_dir,
        seed=args.seed,
    )
    coordinator.run()


if __name__ == "__main__":
    main()
