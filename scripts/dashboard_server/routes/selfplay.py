from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from ..helpers import PIPELINE_DIR


router = APIRouter()


def _history_path() -> Path:
    return PIPELINE_DIR / "selfplay_history.json"


def _load_history() -> List[Dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to read self-play history: {exc}")
    raise HTTPException(status_code=500, detail="Invalid self-play history format; expected a list")


@router.get("/api/selfplay/history")
def get_selfplay_history(limit: int = Query(default=25, ge=1, le=200)) -> Dict[str, Any]:
    entries = _load_history()
    if not entries:
        return {"history": []}
    # Ensure chronological order (newest first)
    entries_sorted = sorted(entries, key=lambda item: (item.get("timestamp"), item.get("iteration", -1)))
    return {"history": entries_sorted[-limit:][::-1]}


@router.get("/api/selfplay/history/latest")
def get_latest_selfplay() -> Dict[str, Any]:
    entries = _load_history()
    if not entries:
        return {"entry": None}
    latest = max(entries, key=lambda item: (item.get("timestamp"), item.get("iteration", -1)))
    return {"entry": latest}
