from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .helpers import PIPELINE_DIR, ROOT


ML_RUNS_DIR = PIPELINE_DIR / "ml_runs"

_CODENAME_RNG = random.SystemRandom()
_CODENAME_FIRST_NAMES = [
    "Lucia",
    "Mateo",
    "Sofia",
    "Diego",
    "Ines",
    "Carlos",
    "Harper",
    "Ethan",
    "Ava",
    "Noah",
    "Amelia",
    "Oliver",
    "Zofia",
    "Jakub",
    "Maja",
    "Leon",
    "Ania",
    "Oskar",
    "Hugo",
    "Manon",
    "Leo",
    "Camille",
    "Lucas",
    "Chloe",
]
_CODENAME_LAST_NAMES = [
    "Garcia",
    "Lopez",
    "Martinez",
    "Santos",
    "Hernandez",
    "Ramirez",
    "Smith",
    "Johnson",
    "Bennett",
    "Clark",
    "Baker",
    "Turner",
    "Kowalski",
    "Nowak",
    "Wisniewski",
    "Lewandowski",
    "Mazur",
    "Kaminski",
    "Dubois",
    "Lefevre",
    "Moreau",
    "Rousseau",
    "Leroux",
    "Boulanger",
]
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class MLLabRunPlan:
    run_dir: Path
    run_dir_label: str
    spec_path: Path
    seed_label: int
    dataset_label: str
    run_stamp: str


def generate_ml_lab_codename() -> str:
    first = _CODENAME_RNG.choice(_CODENAME_FIRST_NAMES)
    last = _CODENAME_RNG.choice(_CODENAME_LAST_NAMES)
    return f"{first}-{last}"


def slugify_ml_lab_label(value: str) -> str:
    cleaned = _SLUG_RE.sub("_", value.strip()).strip("_")
    return cleaned or "custom"


def normalize_ml_lab_seed(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 42


def ensure_ml_lab_runs_dir(runs_dir: Path = ML_RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def safe_read_ml_lab_json(path: Path) -> Any | None:
    try:
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def format_ml_lab_path_for_ui(path: Path, *, root_dir: Path = ROOT, pipeline_dir: Path = PIPELINE_DIR) -> str:
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        try:
            return str(path.relative_to(pipeline_dir))
        except ValueError:
            return str(path)


def find_ml_lab_runs(*, runs_dir: Path = ML_RUNS_DIR) -> list[Path]:
    runs = [path for path in ensure_ml_lab_runs_dir(runs_dir).glob("*") if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return runs


def resolve_ml_lab_run_dir(
    run_dir: str,
    *,
    root_dir: Path = ROOT,
    runs_dir: Path = ML_RUNS_DIR,
) -> Path:
    candidate = Path(run_dir)
    base = ensure_ml_lab_runs_dir(runs_dir).resolve()
    root = root_dir.resolve()

    def _is_under(path: Path) -> bool:
        try:
            path.relative_to(base)
        except ValueError:
            return False
        return True

    resolved: Path | None = None
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved_root = (root / candidate).resolve()
        if _is_under(resolved_root):
            resolved = resolved_root
        else:
            resolved_base = (base / candidate).resolve()
            if _is_under(resolved_base):
                resolved = resolved_base

    if resolved is None or not _is_under(resolved):
        raise ValueError("run_dir must resolve under pipeline_runs_cs/ml_runs")
    if not resolved.exists():
        raise ValueError("run_dir not found")
    return resolved


def plan_ml_lab_run(
    *,
    payload: dict[str, Any],
    dataset: str,
    cfg_path: str | None,
    now: datetime | None = None,
    runs_dir: Path = ML_RUNS_DIR,
    codename_factory: Callable[[], str] = generate_ml_lab_codename,
) -> MLLabRunPlan:
    run_stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    seed_label = normalize_ml_lab_seed(payload.get("seed"))
    dataset_label = dataset or ""
    if not dataset_label and cfg_path:
        dataset_label = Path(cfg_path).stem
    dataset_label = slugify_ml_lab_label(dataset_label)

    base_dir = ensure_ml_lab_runs_dir(runs_dir)
    attempts = 0
    while True:
        codename = codename_factory()
        if attempts:
            codename = f"{codename}-{attempts}"
        run_dir = base_dir / f"run_ml_{codename}_seed{seed_label}_{dataset_label}_{run_stamp}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            return MLLabRunPlan(
                run_dir=run_dir,
                run_dir_label=format_ml_lab_path_for_ui(run_dir),
                spec_path=run_dir / "ml_spec.json",
                seed_label=seed_label,
                dataset_label=dataset_label,
                run_stamp=run_stamp,
            )
        attempts += 1
