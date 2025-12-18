from __future__ import annotations
from typing import Any, Dict, List

_GEN_DIAGNOSTICS: List[Dict[str, Any]] = []


def reset() -> None:
    """Clear all recorded diagnostics."""
    _GEN_DIAGNOSTICS.clear()


def record_generation(
    *,
    generation: int,
    eval_stats: Dict[str, Any],
    eval_events: List[Dict[str, Any]] | None = None,
    best: Dict[str, Any] | None = None,
) -> None:
    """Record one generation snapshot.

    Required core fields are normalized; optional sections may be empty.
    """
    _GEN_DIAGNOSTICS.append(
        {
            "generation": int(generation),
            "eval_stats": dict(eval_stats or {}),
            "events_sample": list(eval_events or []),
            "best": dict(best or {}),
        }
    )


def enrich_last(**extras: Any) -> None:
    """Merge additional fields into the last recorded generation entry."""
    if not _GEN_DIAGNOSTICS:
        return
    _GEN_DIAGNOSTICS[-1].update(extras)


def get_all() -> List[Dict[str, Any]]:
    """Return a deep-ish copy of all diagnostics for serialization."""
    # Shallow copy of top-level dicts is enough for JSON serialization stability
    return [dict(x) for x in _GEN_DIAGNOSTICS]
