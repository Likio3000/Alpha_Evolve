from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from ..helpers import ROOT

try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # pragma: no cover
    _toml = None  # type: ignore


router = APIRouter()


def _configs_dir() -> Path:
    return ROOT / "configs"


def _safe_config_path(name: str) -> Path:
    # Normalize to filename only, enforce .toml
    base = Path(name).name
    if not base.endswith(".toml"):
        base += ".toml"
    p = _configs_dir() / base
    # Ensure path is within configs dir
    try:
        p.resolve().relative_to(_configs_dir().resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return p


def _load_toml(path: Path) -> Dict[str, Any]:
    if _toml is None:
        raise HTTPException(status_code=500, detail="TOML parser not available")
    try:
        with path.open("rb") as fh:
            return _toml.load(fh)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Config not found: {path}")
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Failed to parse TOML: {e}")


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


@router.get("/api/config/defaults")
def get_defaults() -> Dict[str, Any]:
    """Return default parameter values and known choice lists.

    Defaults are read from configs/all_params.toml if present.
    """
    cfg = _configs_dir() / "all_params.toml"
    evo: Dict[str, Any] = {}
    bt: Dict[str, Any] = {}
    if cfg.exists():
        data = _load_toml(cfg)
        evo = dict(data.get("evolution", {}))
        bt = dict(data.get("backtest", {}))
    choices = {
        # Shared/select-type choices
        "max_lookback_data_option": ["common_1200", "specific_long_10k", "full_overlap"],
        "selection_metric": ["ramped", "fixed", "ic", "auto", "phased"],
        "split_weighting": ["equal", "by_points"],
        "scale": ["zscore", "rank", "sign", "madz", "winsor"],
        "hof_corr_mode": ["flat", "per_bar"],
    }
    return {"evolution": evo, "backtest": bt, "choices": choices}


@router.get("/api/config/list")
def list_configs() -> Dict[str, Any]:
    items = []
    for p in sorted(_configs_dir().glob("*.toml")):
        items.append({"name": p.name, "path": str(p.relative_to(ROOT))})
    return {"items": items}


def _preset_map() -> Dict[str, str]:
    presets: Dict[str, str] = {}
    cand = {
        "crypto": "configs/crypto.toml",
        "crypto_4h_fast": "configs/crypto_4h_fast.toml",
        "sp500": "configs/sp500.toml",
    }
    for key, rel in cand.items():
        if (ROOT / rel).exists():
            presets[key] = rel
    return presets


@router.get("/api/config/presets")
def get_presets() -> Dict[str, Any]:
    return {"presets": _preset_map()}


@router.get("/api/config/preset-values")
def get_preset_values(dataset: Optional[str] = Query(default=None), path: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    if not dataset and not path:
        raise HTTPException(status_code=400, detail="Provide dataset or path")
    target: Optional[Path] = None
    if dataset:
        rel = _preset_map().get(dataset.strip().lower())
        if not rel:
            raise HTTPException(status_code=404, detail="Unknown dataset preset")
        target = ROOT / rel
    elif path:
        # Normalize and ensure under ROOT/configs
        p = Path(path)
        if p.is_absolute():
            raise HTTPException(status_code=400, detail="Path must be relative")
        cand = (ROOT / p).resolve()
        try:
            cand.relative_to(_configs_dir().resolve())
        except Exception:
            raise HTTPException(status_code=400, detail="Path must be under configs/")
        target = cand
    if target is None:
        raise HTTPException(status_code=400, detail="Unable to resolve preset path")
    data = _load_toml(target)
    return {"evolution": data.get("evolution", {}), "backtest": data.get("backtest", {})}


@router.post("/api/config/save")
def save_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Missing name")
    evo = payload.get("evolution") or {}
    bt = payload.get("backtest") or {}
    if not isinstance(evo, dict) or not isinstance(bt, dict):
        raise HTTPException(status_code=400, detail="Invalid evolution/backtest")
    out_path = _safe_config_path(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    txt = _to_toml({"evolution": evo, "backtest": bt})
    out_path.write_text(txt, encoding="utf-8")
    return {"saved": str(out_path.relative_to(ROOT))}

