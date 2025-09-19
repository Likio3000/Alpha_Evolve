from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from alpha_framework.alpha_framework_program import AlphaProgram
from .evaluation_logic import EvalResult


@dataclass
class QDArchiveEntry:
    fingerprint: str
    generation: int
    descriptors: Tuple[int, int]
    metrics: EvalResult
    program: AlphaProgram


class QualityDiversityArchive:
    """Simple MAP-Elites style archive keyed by descriptor bins.

    Descriptors:
        0) Turnover proxy bucket
        1) Complexity bucket (normalized program size)
    """

    def __init__(
        self,
        *,
        turnover_bins: Iterable[float],
        complexity_bins: Iterable[float],
        max_entries: int = 256,
    ) -> None:
        self.turnover_bins = tuple(sorted(turnover_bins))
        self.complexity_bins = tuple(sorted(complexity_bins))
        self.max_entries = max_entries
        self._cells: Dict[Tuple[int, int], QDArchiveEntry] = {}

    def clear(self) -> None:
        self._cells.clear()

    def _bin_index(self, value: float, edges: Tuple[float, ...]) -> int:
        # Edges include upper bounds. Values beyond last edge land in final bin.
        for idx, edge in enumerate(edges):
            if value <= edge:
                return idx
        return len(edges)

    def _descriptor(self, metrics: EvalResult, prog: AlphaProgram, max_ops: int) -> Tuple[int, int]:
        turnover = float(metrics.turnover_proxy)
        size = int(getattr(prog, "size", 0) or 0)
        max_ops = max(1, max_ops)
        complexity_norm = size / max_ops
        turnover_idx = self._bin_index(turnover, self.turnover_bins)
        complexity_idx = self._bin_index(complexity_norm, self.complexity_bins)
        return turnover_idx, complexity_idx

    def add(
        self,
        *,
        prog: AlphaProgram,
        metrics: EvalResult,
        generation: int,
        max_ops: int,
    ) -> Optional[QDArchiveEntry]:
        desc = self._descriptor(metrics, prog, max_ops)
        fp = getattr(prog, "fingerprint", None)
        if fp is None:
            return None
        candidate = QDArchiveEntry(
            fingerprint=fp,
            generation=generation,
            descriptors=desc,
            metrics=metrics,
            program=prog,
        )
        existing = self._cells.get(desc)
        # Prefer higher fitness
        if existing is None or metrics.fitness > existing.metrics.fitness:
            self._cells[desc] = candidate
            # Enforce capacity by dropping lowest-fitness entry globally
            if len(self._cells) > self.max_entries:
                worst_key = min(
                    self._cells.keys(),
                    key=lambda k: self._cells[k].metrics.fitness,
                )
                if worst_key != desc:
                    self._cells.pop(worst_key, None)
            return candidate
        return None

    def elites(self) -> List[QDArchiveEntry]:
        return sorted(
            self._cells.values(),
            key=lambda e: e.metrics.fitness,
            reverse=True,
        )

    def summary(self) -> Dict[str, object]:
        elites = self.elites()
        return {
            "cells": len(self._cells),
            "turnover_bins": self.turnover_bins,
            "complexity_bins": self.complexity_bins,
            "elites": [
                {
                    "fingerprint": e.fingerprint,
                    "generation": e.generation,
                    "descriptors": e.descriptors,
                    "fitness": float(e.metrics.fitness),
                    "mean_ic": float(e.metrics.mean_ic),
                    "turnover": float(e.metrics.turnover_proxy),
                    "ops": getattr(e.program, "size", 0),
                }
                for e in elites[:50]
            ],
        }


_ARCHIVE: Optional[QualityDiversityArchive] = None


def initialize_archive(
    *,
    turnover_bins: Iterable[float],
    complexity_bins: Iterable[float],
    max_entries: int,
) -> None:
    global _ARCHIVE
    _ARCHIVE = QualityDiversityArchive(
        turnover_bins=turnover_bins,
        complexity_bins=complexity_bins,
        max_entries=max_entries,
    )


def clear_archive() -> None:
    if _ARCHIVE is not None:
        _ARCHIVE.clear()


def add_candidate(
    prog: AlphaProgram,
    metrics: EvalResult,
    generation: int,
    max_ops: int,
) -> Optional[QDArchiveEntry]:
    if _ARCHIVE is None:
        return None
    return _ARCHIVE.add(prog=prog, metrics=metrics, generation=generation, max_ops=max_ops)


def get_elites() -> List[QDArchiveEntry]:
    if _ARCHIVE is None:
        return []
    return _ARCHIVE.elites()


def get_summary() -> Dict[str, object]:
    if _ARCHIVE is None:
        return {"cells": 0, "elites": []}
    return _ARCHIVE.summary()
