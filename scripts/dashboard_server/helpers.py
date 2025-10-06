from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import re
import json
import time
from queue import Empty

from django.http import StreamingHttpResponse


# Resolve project root relative to this file (under scripts/dashboard_server)
ROOT: Path = Path(__file__).resolve().parents[2]


def _compute_pipeline_dir() -> Path:
    override = os.environ.get("AE_PIPELINE_DIR") or os.environ.get("AE_OUTPUT_DIR")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate
    return (ROOT / "pipeline_runs_cs").resolve()


PIPELINE_DIR: Path = _compute_pipeline_dir()


def read_best_sharpe_from_run(run_dir: Path) -> Optional[float]:
    bt_dir = run_dir / "backtest_portfolio_csvs"
    candidates = sorted(bt_dir.glob("backtest_summary_top*.csv"))
    if not candidates:
        return None
    csv_path = candidates[-1]
    try:
        import csv
        best = None
        with open(csv_path, newline="") as fh:
            rdr = csv.DictReader(fh)
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


def resolve_latest_run_dir() -> Optional[Path]:
    latest = PIPELINE_DIR / "LATEST"
    try:
        if latest.exists():
            p = latest.read_text().strip()
            if p:
                raw_path = Path(p)
                candidates = []
                if raw_path.is_absolute():
                    candidates.append(raw_path.resolve())
                else:
                    candidates.append((ROOT / raw_path).resolve())
                    parts = raw_path.parts
                    if not parts or parts[0] != PIPELINE_DIR.name:
                        candidates.append((PIPELINE_DIR / raw_path).resolve())
                for run_path in candidates:
                    if run_path.exists():
                        return run_path
    except Exception:
        return None
    return None


def build_pipeline_args(payload: Dict[str, Any], include_runner: bool = True) -> list[str]:
    """Map JSON payload to a run_pipeline invocation args list.

    When ``include_runner`` is ``True`` (default) the returned list is suitable for
    spawning via ``uv run run_pipeline.py``. When ``False`` the leading launcher
    tokens are omitted, making the list suitable for direct argument parsing
    (e.g., programmatic consumption).
    """
    gens = int(payload.get("generations", 5))
    args: list[str] = []
    if include_runner:
        args.extend(["uv", "run", "run_pipeline.py"])
    args.append(str(gens))
    dataset = str(payload.get("dataset", "")).strip().lower()
    cfg_path = payload.get("config")
    if not cfg_path and dataset:
        if dataset in ("crypto", "crypto_4h", "crypto4h"):
            cfg_path = str(ROOT / "configs" / "crypto_4h_fast.toml")
        elif dataset in ("sp500", "s&p500", "snp500"):
            cfg_path = str(ROOT / "configs" / "sp500.toml")
    if cfg_path:
        args += ["--config", str(cfg_path)]
    if payload.get("data_dir"):
        args += ["--data_dir", str(payload["data_dir"])]
    raw_overrides = payload.get("overrides") or {}
    overrides = dict(raw_overrides)
    try:
        if "generations" in overrides:
            gens = int(overrides.pop("generations"))
    except Exception:
        pass
    overrides.pop("sector_mapping", None)
    reserved = {"generations", "dataset", "config", "data_dir", "overrides"}
    for k, v in payload.items():
        if k in reserved:
            continue
        if isinstance(v, (str, int, float, bool)):
            overrides[k] = v
    for k, v in overrides.items():
        if not isinstance(v, (str, int, float, bool)):
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
        else:
            args += [flag, str(v)]
    return args


# Shared regexes for log parsing
RE_CANDIDATE = re.compile(r"^→ Candidate\s+(\d+)/(\d+):\s+(.*)$")
RE_SHARPE = re.compile(r"Sharpe\(best\)\s*=\s*([+\-]?[0-9.]+)")
RE_DIAG = re.compile(r"DIAG\s+(\{.*\})$")
RE_PROGRESS = re.compile(r"PROGRESS\s+(\{.*\})$")


def make_sse_response(queue, keepalive_seconds: float = 10.0) -> StreamingHttpResponse:
    def _event_stream():
        while True:
            try:
                item = queue.get(timeout=keepalive_seconds)
                yield f"data: {item}\n\n"
            except Empty:
                payload = json.dumps({"t": time.time()})
                yield f"event: ping\ndata: {payload}\n\n"

    response = StreamingHttpResponse(_event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response
