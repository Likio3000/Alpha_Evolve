from __future__ import annotations

import json
import math
import re
from typing import Any

from .helpers import RE_DIAG, RE_PROGRESS, RE_SHARPE


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ARTEFACTS_PATH_RE = re.compile(r"artefacts in\s+(?P<path>.+)$", re.IGNORECASE)
OUTPUT_PATH_RE = re.compile(r"outputs?\s*(?:→|->)\s*(?P<path>.+)$", re.IGNORECASE)


def parse_json_constant(token: str) -> float:
    if token == "NaN":
        return float("nan")
    if token == "Infinity":
        return float("inf")
    if token == "-Infinity":
        return float("-inf")
    raise ValueError(f"Unexpected JSON constant: {token}")


def sanitize_json_data(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: sanitize_json_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_data(item) for item in value]
    return value


def line_to_pipeline_event(line: str) -> dict[str, Any]:
    line = line.rstrip("\n")
    if line:
        line = ANSI_ESCAPE_RE.sub("", line)
    if not line:
        return {"type": "log", "raw": ""}
    if (match := RE_DIAG.search(line)) is not None:
        try:
            data = json.loads(match.group(1), parse_constant=parse_json_constant)
            data = sanitize_json_data(data)
            return {"type": "diag", "data": data, "raw": line}
        except Exception:
            return {"type": "log", "raw": line}
    if (match := RE_PROGRESS.search(line)) is not None:
        try:
            data = json.loads(match.group(1), parse_constant=parse_json_constant)
            data = sanitize_json_data(data)
            subtype = data.get("type") if isinstance(data, dict) else None
            if subtype == "gen_summary":
                return {"type": "gen_summary", "data": data, "raw": line}
            event: dict[str, Any] = {"type": "progress", "data": data, "raw": line}
            if isinstance(subtype, str):
                event["subtype"] = subtype
            return event
        except Exception:
            return {"type": "log", "raw": line}
    if (match := RE_SHARPE.search(line)) is not None:
        try:
            return {"type": "score", "sharpe_best": float(match.group(1)), "raw": line}
        except Exception:
            return {"type": "log", "raw": line}
    return {"type": "log", "raw": line}


def resolve_run_dir_hint(line: str) -> str | None:
    if not line:
        return None
    match = ARTEFACTS_PATH_RE.search(line)
    if match is not None:
        path = match.group("path").strip()
        return path or None
    match = OUTPUT_PATH_RE.search(line)
    if match is not None:
        path = match.group("path").strip()
        return path or None
    return None
