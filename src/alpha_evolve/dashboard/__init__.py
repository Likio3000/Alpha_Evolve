"""
Dashboard server and UI integration for Alpha Evolve.
"""

from .api.app import create_app, app

__all__ = [
    "create_app",
    "app",
]
