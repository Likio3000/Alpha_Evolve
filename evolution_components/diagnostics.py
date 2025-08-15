from __future__ import annotations
from typing import Any, Dict, List

_GEN_DIAGNOSTICS: List[Dict[str, Any]] = []

def reset() -> None:
    _GEN_DIAGNOSTICS.clear()

def record_generation(
    generation: int,
    eval_stats: Dict[str, Any],
    eval_events: List[Dict[str, Any]],
    best: Dict[str, Any],
) -> None:
    _GEN_DIAGNOSTICS.append({
        "generation": generation,
        "eval_stats": eval_stats,
        "events_sample": eval_events,
        "best": best,
    })

def get_all() -> List[Dict[str, Any]]:
    return list(_GEN_DIAGNOSTICS)

