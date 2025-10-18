#!/usr/bin/env python3
from __future__ import annotations

"""
run_dashboard.py – Launcher for the Dashboard API (unified logging).

Usage:
  uv run scripts/run_dashboard.py

Environment variables:
  LOG_LEVEL=DEBUG|INFO|WARNING|ERROR (default: INFO)
  LOG_FILE=path/to/file.log (optional)
  ACCESS_LOG=0|1 (uvicorn access log; default: 0)
  HOST=127.0.0.1 (default)
  PORT=8000 (default)
"""

import logging
import os
from pathlib import Path
import sys

# Ensure project root is on sys.path so top-level imports work when run as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import uvicorn
from alpha_evolve.dashboard.api.app import app
from alpha_evolve.utils.logging import setup_logging


def _level_from_env(var: str, default: int = logging.INFO) -> int:
    val = os.environ.get(var)
    if not val:
        return default
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    return mapping.get(val.upper(), default)


def main() -> None:
    level = _level_from_env("LOG_LEVEL", logging.INFO)
    log_file = os.environ.get("LOG_FILE")
    log_config = setup_logging(level=level, log_file=log_file)

    host = os.environ.get("HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("PORT", "8000"))
    except Exception:
        port = 8000
    access_log = os.environ.get("ACCESS_LOG", "0") in ("1", "true", "True")

    logger = logging.getLogger("dashboard.server")
    ui_url = f"http://{host}:{port}/ui"
    logger.info("Dashboard ready → %s", ui_url)
    logger.debug("Host=%s | Port=%d | Access log=%s", host, port, "on" if access_log else "off")

    uvicorn.run(
        app,
        host=host,
        port=port,
        access_log=access_log,
        log_level=logging.getLevelName(level).lower(),
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
