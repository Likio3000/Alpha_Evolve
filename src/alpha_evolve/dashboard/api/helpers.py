from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import fields as dc_fields
import re
import json
import time
from queue import Empty
import asyncio
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from alpha_evolve.config.model import EvolutionConfig, BacktestConfig
from alpha_evolve.utils.run_artifacts import (
    resolve_latest_run_dir as resolve_latest_pipeline_run_dir,
    select_backtest_summary_csv,
)

from django.http import HttpResponse, StreamingHttpResponse


# Resolve project root relative to this file (now within src/alpha_evolve/dashboard/api)
ROOT: Path = Path(__file__).resolve().parents[4]


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
    csv_path = select_backtest_summary_csv(bt_dir)
    if csv_path is None:
        return None
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
    return resolve_latest_pipeline_run_dir(PIPELINE_DIR, project_root=ROOT)


def resolve_config_path(config_path: str | os.PathLike[str]) -> Optional[Path]:
    raw_path = Path(config_path).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path.resolve())
    else:
        candidates.append((ROOT / raw_path).resolve())
        candidates.append((Path.cwd() / raw_path).resolve())
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


_DATASET_PRESETS: Dict[str, Path] = {
    "sp500": ROOT / "configs" / "sp500.toml",
    "sp500_small": ROOT / "configs" / "sp500_small.toml",
}
_DATASET_ALIASES: Dict[str, str] = {
    "s&p500": "sp500",
    "snp500": "sp500",
    "sp500-small": "sp500_small",
    "sp500subset": "sp500_small",
}

_CLI_KEY_ALIASES: Dict[str, str] = {
    # Backward-compatible alias from legacy dashboard payloads.
    "bt_top": "top_to_backtest",
}


def _collect_negatable_bool_flags() -> set[str]:
    names: set[str] = set()
    for dc in (EvolutionConfig, BacktestConfig):
        try:
            defaults = dc()
        except Exception:
            continue
        for f in dc_fields(dc):
            if f.type is bool and getattr(defaults, f.name, False) is True:
                names.add(f.name)
    # Pipeline-only bools with explicit --no-* support.
    names.update({"persist_hof_per_gen", "diagnostics_plots", "backtest_plots"})
    return names


_NEGATABLE_BOOL_FLAGS: set[str] = _collect_negatable_bool_flags()


def resolve_dataset_preset(name: str) -> Optional[Path]:
    key = name.strip().lower()
    canonical = _DATASET_ALIASES.get(key, key)
    path = _DATASET_PRESETS.get(canonical)
    if path and path.exists():
        return path
    return None


def dataset_presets() -> Dict[str, Path]:
    return {key: path for key, path in _DATASET_PRESETS.items() if path.exists()}


def known_dataset_names() -> set[str]:
    names = set(_DATASET_PRESETS.keys())
    names.update(_DATASET_ALIASES.keys())
    return names


def build_pipeline_args(
    payload: Dict[str, Any], include_runner: bool = True
) -> list[str]:
    """Map JSON payload to a run_pipeline invocation args list.

    When ``include_runner`` is ``True`` (default) the returned list is suitable for
    spawning via ``uv run python -m alpha_evolve.cli.pipeline``. When ``False`` the leading launcher
    tokens are omitted, making the list suitable for direct argument parsing
    (e.g., programmatic consumption).
    """
    gens = int(payload.get("generations", 5))
    raw_overrides = payload.get("overrides") or {}
    overrides: Dict[str, Any] = {}
    if isinstance(raw_overrides, dict):
        for k, v in raw_overrides.items():
            key = _CLI_KEY_ALIASES.get(str(k), str(k))
            overrides[key] = v
    try:
        if "generations" in overrides:
            gens = int(overrides.pop("generations"))
    except Exception:
        pass

    args: list[str] = []
    if include_runner:
        args.extend(["uv", "run", "python", "-m", "alpha_evolve.cli.pipeline"])
    args.append(str(gens))
    dataset = str(payload.get("dataset", "")).strip().lower()
    cfg_path = payload.get("config")
    if not cfg_path and dataset:
        preset = resolve_dataset_preset(dataset)
        if preset:
            cfg_path = str(preset)
    if cfg_path:
        args += ["--config", str(cfg_path)]
    if payload.get("data_dir"):
        args += ["--data_dir", str(payload["data_dir"])]
    overrides.pop("sector_mapping", None)
    reserved = {
        "generations",
        "dataset",
        "config",
        "data_dir",
        "overrides",
        "runner_mode",
        "job_runner",
    }
    for k, v in payload.items():
        if k in reserved:
            continue
        key = _CLI_KEY_ALIASES.get(str(k), str(k))
        if isinstance(v, (str, int, float, bool)):
            overrides[key] = v
    for k, v in overrides.items():
        if not isinstance(v, (str, int, float, bool)):
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
            elif k in _NEGATABLE_BOOL_FLAGS:
                args.append(f"--no-{k}")
        else:
            args += [flag, str(v)]
    return args


# Shared regexes for log parsing
RE_CANDIDATE = re.compile(r"^→ Candidate\s+(\d+)/(\d+):\s+(.*)$")
RE_SHARPE = re.compile(r"Sharpe\(best\)\s*=\s*([+\-]?[0-9.]+)")
RE_DIAG = re.compile(r"DIAG\s+(\{.*\})$")
RE_PROGRESS = re.compile(r"PROGRESS\s+(\{.*\})$")


class _SSEStream:
    def __init__(self, q, keepalive: float, *, prefer_async: bool = True) -> None:
        self._queue = q
        self._keepalive = keepalive
        self._prefer_async = prefer_async

    @staticmethod
    def _ping() -> str:
        payload = json.dumps({"t": time.time()})
        return f"event: ping\ndata: {payload}\n\n"

    def __iter__(self):
        if self._prefer_async:
            raise TypeError("SSEStream prefers async iteration under ASGI.")
        # Support synchronous iteration for environments that explicitly opt-in.
        while True:
            try:
                item = self._queue.get(timeout=self._keepalive)
                yield f"data: {item}\n\n"
            except Empty:
                yield self._ping()

    async def __aiter__(self):
        while True:
            try:
                item = await asyncio.to_thread(self._queue.get, True, self._keepalive)
                yield f"data: {item}\n\n"
            except Empty:
                yield self._ping()


def make_sse_response(queue, keepalive_seconds: float = 10.0) -> StreamingHttpResponse:
    stream = _SSEStream(queue, keepalive_seconds)
    # Django's ASGI handler will treat objects that also implement __iter__ as synchronous
    # iterables, so pass the async generator directly to preserve streaming semantics.
    response = StreamingHttpResponse(
        stream.__aiter__(), content_type="text/event-stream"
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def _content_type_for_path(path: Path) -> str:
    ctype, _ = mimetypes.guess_type(str(path))
    if not ctype:
        return "application/octet-stream"
    if ctype.startswith("text/") and "charset=" not in ctype:
        return f"{ctype}; charset=utf-8"
    if ctype in {"application/javascript", "application/json"}:
        return f"{ctype}; charset=utf-8"
    return ctype


def file_response(
    request_method: str,
    path: Path,
    *,
    content_type: str | None = None,
    content_disposition: str = "inline",
    filename: str | None = None,
    chunk_size: int = 64 * 1024,
) -> HttpResponse:
    """Return an ASGI-friendly file response without Django's sync-stream buffering warnings."""

    if request_method.upper() == "HEAD":
        resp = HttpResponse(
            b"", content_type=content_type or _content_type_for_path(path)
        )
        try:
            resp["Content-Length"] = str(path.stat().st_size)
        except Exception:
            pass
        return resp

    async def _stream() -> Any:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        f = None
        try:
            f = await loop.run_in_executor(executor, open, path, "rb")
            while True:
                chunk = await loop.run_in_executor(executor, f.read, int(chunk_size))
                if not chunk:
                    break
                yield chunk
        finally:
            if f is not None:
                try:
                    await asyncio.shield(loop.run_in_executor(executor, f.close))
                except Exception:
                    pass
            executor.shutdown(wait=False)

    resp = StreamingHttpResponse(
        _stream(), content_type=content_type or _content_type_for_path(path)
    )
    try:
        resp["Content-Length"] = str(path.stat().st_size)
    except Exception:
        pass
    if filename is None:
        filename = path.name
    resp["Content-Disposition"] = f'{content_disposition}; filename="{filename}"'
    return resp
