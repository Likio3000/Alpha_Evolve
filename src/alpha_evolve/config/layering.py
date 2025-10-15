"""Layer configuration values from files, environment variables, and CLI arguments, letting later sources override earlier ones."""

from __future__ import annotations
from dataclasses import fields as dc_fields
from typing import Any, Dict, Tuple, Type
import os


def _try_parse_bool(s: str) -> bool:
    t = s.strip().lower()
    if t in {"1", "true", "yes", "on", "y"}:
        return True
    if t in {"0", "false", "no", "off", "n"}:
        return False
    # Fallback: non-empty truthy
    return bool(t)


def _coerce_value(val: Any, to_type: type) -> Any:
    if isinstance(val, to_type) or val is None:
        return val
    try:
        if to_type is bool:
            if isinstance(val, str):
                return _try_parse_bool(val)
            return bool(val)
        if to_type in (int, float, str):
            return to_type(val)
    except Exception:
        pass
    return val


def _load_toml(path: str) -> Dict[str, Any]:
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("YAML config requested but PyYAML is not installed") from e
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a config file (TOML or YAML).

    Returns a nested dict. Accepts top-level keys like 'evolution' and 'backtest'
    or flat keys matching dataclass field names.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".toml", ".tml"):
        return _load_toml(path)
    if ext in (".yaml", ".yml"):
        return _load_yaml(path)
    raise RuntimeError(f"Unsupported config extension: {ext}")


def _flatten_sectioned_config(raw: Dict[str, Any], section: str | None) -> Dict[str, Any]:
    if section and section in raw and isinstance(raw[section], dict):
        return dict(raw.get(section, {}))
    # If not sectioned, return a shallow copy of top-level
    return dict(raw)


def _collect_env_overrides(dc_type: Type, prefixes: Tuple[str, ...]) -> Dict[str, Any]:
    """Collect env var overrides for a dataclass.

    Env keys use uppercase with underscores, e.g. AE_DATA_DIR or AE_BT_DATA_DIR.
    Later prefixes in the tuple have higher precedence.
    """
    out: Dict[str, Any] = {}
    # Build mapping of env name -> field name
    fields = list(dc_fields(dc_type))
    name_map = {f.name.lower(): f for f in fields}
    for prefix in prefixes:
        plen = len(prefix)
        for k, v in os.environ.items():
            if not k.startswith(prefix):
                continue
            key = k[plen:].lower()
            if key in name_map:
                f = name_map[key]
                out[f.name] = _coerce_value(v, f.type)
    return out


def layer_dataclass_config(
    dc_type: Type,
    *,
    file_cfg: Dict[str, Any] | None,
    env_prefixes: Tuple[str, ...],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a merged mapping for dc_type following precedence: file < env < CLI.

    Only fields present on the dataclass are included; types are coerced best-effort.
    """
    result: Dict[str, Any] = {}
    field_types = {f.name: f.type for f in dc_fields(dc_type)}
    # 1) file
    if file_cfg:
        for k, v in file_cfg.items():
            if k in field_types:
                result[k] = _coerce_value(v, field_types[k])
    # 2) env
    env = _collect_env_overrides(dc_type, env_prefixes)
    for k, v in env.items():
        result[k] = _coerce_value(v, field_types[k])
    # 3) CLI
    for k, v in cli_overrides.items():
        if k in field_types:
            result[k] = _coerce_value(v, field_types[k])
    return result

