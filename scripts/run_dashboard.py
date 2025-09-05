#!/usr/bin/env python3
from __future__ import annotations

"""
run_dashboard.py – Thin launcher for the Iterative Dashboard API.

Preferred entrypoint:
  uv run scripts/run_dashboard.py
"""

import uvicorn
from scripts.dashboard_server.app import app


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)


if __name__ == "__main__":
    main()
