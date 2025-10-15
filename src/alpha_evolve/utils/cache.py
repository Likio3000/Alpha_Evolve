from __future__ import annotations
import hashlib
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .data_loading import DataBundle


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _cache_dir() -> Path:
    p = os.environ.get("AE_ALIGN_CACHE_DIR")
    if p:
        return Path(p)
    return Path.cwd() / ".cache" / "align_cache"


def compute_align_cache_key(
    *,
    data_dir: str,
    feature_fn_name: str,
    strategy: str,
    min_common_points: int,
    eval_lag: int,
    include_lag_in_required_length: bool,
    fixed_trim_include_lag: bool,
) -> str:
    """Compute a stable cache key from file metadata and parameters.

    Uses filenames, sizes, mtimes (rounded to seconds) and parameters. We avoid
    content hashing for speed; for correctness, users should clear cache if
    content changes without mtime update (rare).
    """
    h = hashlib.sha1()
    h.update(f"feat:{feature_fn_name}|strat:{strategy}|min:{min_common_points}|lag:{eval_lag}|incl:{int(include_lag_in_required_length)}|trim:{int(fixed_trim_include_lag)}".encode())
    try:
        for name in sorted(os.listdir(data_dir)):
            if not name.lower().endswith(".csv"):
                continue
            p = Path(data_dir) / name
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            h.update(name.encode())
            h.update(str(st.st_size).encode())
            h.update(str(int(st.st_mtime)).encode())
    except Exception:
        pass
    return h.hexdigest()


def load_aligned_bundle_from_cache(key: str) -> Optional[DataBundle]:
    if _truthy_env("AE_DISABLE_ALIGN_CACHE"):
        return None
    cd = _cache_dir()
    fp = cd / f"{key}.pkl"
    if not fp.exists():
        return None
    try:
        with open(fp, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def save_aligned_bundle_to_cache(key: str, bundle: DataBundle) -> Optional[Path]:
    if _truthy_env("AE_DISABLE_ALIGN_CACHE"):
        return None
    cd = _cache_dir()
    try:
        cd.mkdir(parents=True, exist_ok=True)
        fp = cd / f"{key}.pkl"
        with open(fp, "wb") as fh:
            pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return fp
    except Exception:
        return None
