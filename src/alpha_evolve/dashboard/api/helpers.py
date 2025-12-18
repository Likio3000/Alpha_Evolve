from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import re
import json
import time
from queue import Empty
import asyncio
import mimetypes
from concurrent.futures import ThreadPoolExecutor

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
    raw_overrides = payload.get("overrides") or {}
    overrides = dict(raw_overrides)
    try:
        if "generations" in overrides:
            gens = int(overrides.pop("generations"))
    except Exception:
        pass
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
