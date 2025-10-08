from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from django.http import HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from ..helpers import ROOT
from ..http import json_error, json_response

try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # pragma: no cover
    _toml = None  # type: ignore


def _configs_dir() -> Path:
    return ROOT / "configs"


def _safe_config_path(name: str) -> Path:
    base = Path(name).name
    if not base.endswith(".toml"):
        base += ".toml"
    p = _configs_dir() / base
    try:
        p.resolve().relative_to(_configs_dir().resolve())
    except Exception as exc:  # pragma: no cover - guardrail
        raise ValueError("Invalid filename") from exc
    return p


def _load_toml(path: Path) -> Dict[str, Any]:
    if _toml is None:
        raise RuntimeError("TOML parser not available")
    try:
        with path.open("rb") as fh:
            return _toml.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config not found: {path}") from exc
    except Exception as exc:  # pragma: no cover - passthrough parsing errors
        raise ValueError(f"Failed to parse TOML: {exc}") from exc


def _to_toml(obj: Dict[str, Dict[str, Any]]) -> str:
    def _fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v)
        s = s.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'"{s}"'

    lines: list[str] = []
    for section in ("evolution", "backtest"):
        if section not in obj:
            continue
        lines.append(f"[{section}]")
        kvs = obj[section]
        for k in sorted(kvs.keys()):
            lines.append(f"{k} = {_fmt(kvs[k])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@require_GET
def get_defaults(request: HttpRequest):
    cfg = _configs_dir() / "all_params.toml"
    evo: Dict[str, Any] = {}
    bt: Dict[str, Any] = {}
    if cfg.exists():
        try:
            data = _load_toml(cfg)
        except (FileNotFoundError, ValueError, RuntimeError):  # pragma: no cover - unexpected
            data = {}
        evo = dict(data.get("evolution", {}))
        bt = dict(data.get("backtest", {}))
    choices = {
        "max_lookback_data_option": ["common_1200", "specific_long_10k", "full_overlap"],
        "selection_metric": ["ramped", "fixed", "ic", "auto", "phased"],
        "split_weighting": ["equal", "by_points"],
        "scale": ["zscore", "rank", "sign", "madz", "winsor"],
        "hof_corr_mode": ["flat", "per_bar"],
    }
    return json_response({"evolution": evo, "backtest": bt, "choices": choices})


@require_GET
def list_configs(request: HttpRequest):
    items = []
    for p in sorted(_configs_dir().glob("*.toml")):
        items.append({"name": p.name, "path": str(p.relative_to(ROOT))})
    return json_response({"items": items})


def _preset_map() -> Dict[str, str]:
    presets: Dict[str, str] = {}
    cand = {
        "sp500": "configs/sp500.toml",
    }
    for key, rel in cand.items():
        if (ROOT / rel).exists():
            presets[key] = rel
    return presets


@require_GET
def get_presets(request: HttpRequest):
    return json_response({"presets": _preset_map()})


@require_GET
def get_preset_values(request: HttpRequest):
    dataset = request.GET.get("dataset")
    path_param = request.GET.get("path")
    if not dataset and not path_param:
        return json_error("Provide dataset or path", 400)
    target: Optional[Path] = None
    if dataset:
        rel = _preset_map().get(dataset.strip().lower())
        if not rel:
            return json_error("Unknown dataset preset", 404)
        target = ROOT / rel
    elif path_param:
        p = Path(path_param)
        if p.is_absolute():
            return json_error("Path must be relative", 400)
        cand = (ROOT / p).resolve()
        try:
            cand.relative_to(_configs_dir().resolve())
        except Exception:
            return json_error("Path must be under configs/", 400)
        target = cand
    if target is None:
        return json_error("Unable to resolve preset path", 400)
    try:
        data = _load_toml(target)
    except FileNotFoundError as exc:
        return json_error(str(exc), 404)
    except (ValueError, RuntimeError) as exc:
        return json_error(str(exc), 400)
    return json_response({"evolution": data.get("evolution", {}), "backtest": data.get("backtest", {})})


@csrf_exempt
@require_POST
def save_config(request: HttpRequest):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return json_error("Invalid JSON body", 400)
    name = str(payload.get("name") or "").strip()
    if not name:
        return json_error("Missing name", 400)
    evo = payload.get("evolution") or {}
    bt = payload.get("backtest") or {}
    if not isinstance(evo, dict) or not isinstance(bt, dict):
        return json_error("Invalid evolution/backtest", 400)
    try:
        out_path = _safe_config_path(name)
    except ValueError as exc:
        return json_error(str(exc), 400)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    txt = _to_toml({"evolution": evo, "backtest": bt})
    out_path.write_text(txt, encoding="utf-8")
    return json_response({"saved": str(out_path.relative_to(ROOT))})
