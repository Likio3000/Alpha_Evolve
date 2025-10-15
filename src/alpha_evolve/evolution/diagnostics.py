"""Compatibility shim; delegates to utils.diagnostics.

Prefer importing utils.diagnostics directly.
"""

from __future__ import annotations
from typing import Any, Dict, List
from alpha_evolve.utils.diagnostics import (
    reset,
    record_generation,
    enrich_last,
    get_all,
)
