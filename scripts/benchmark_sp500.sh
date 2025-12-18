#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd -P)
root="$here/.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-$root/.uv_cache}"

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/benchmark_sp500.py "$@"
elif [ -x "$root/.venv/bin/python" ]; then
  "$root/.venv/bin/python" scripts/benchmark_sp500.py "$@"
else
  python3 scripts/benchmark_sp500.py "$@"
fi

