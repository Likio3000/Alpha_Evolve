#!/usr/bin/env python3
"""Synchronize requirements.txt with dependencies in pyproject.toml."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
pyproject_path = ROOT / "pyproject.toml"
requirements_path = ROOT / "requirements.txt"

with pyproject_path.open("rb") as f:
    data = tomllib.load(f)

deps = data.get("project", {}).get("dependencies", [])

with requirements_path.open("w") as f:
    for dep in deps:
        f.write(dep + "\n")

print(f"Wrote {len(deps)} dependencies to {requirements_path}")
