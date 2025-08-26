from __future__ import annotations
"""Compatibility shim; delegates to utils.diagnostics.

Prefer importing utils.diagnostics directly.
"""
from typing import Any, Dict, List
from utils.diagnostics import reset as reset  # re-export
from utils.diagnostics import record_generation as record_generation  # re-export
from utils.diagnostics import enrich_last as enrich_last  # re-export
from utils.diagnostics import get_all as get_all  # re-export
