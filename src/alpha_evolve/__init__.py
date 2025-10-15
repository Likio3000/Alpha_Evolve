"""
Core package for Alpha Evolve.

This module exposes convenient aliases for the most common entry points so
external callers do not need to import deep submodules directly.
"""

from importlib import import_module

__all__ = [
    "load_module",
]


def load_module(path: str):
    """Thin wrapper around :func:`importlib.import_module` for dynamic loading."""
    return import_module(path)
