"""Compatibility shim that re-exports diagnostics helpers."""

from __future__ import annotations

from alpha_evolve.utils.diagnostics import enrich_last, get_all, record_generation, reset

__all__ = ["reset", "record_generation", "enrich_last", "get_all"]
