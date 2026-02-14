#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd -P)
root="$here/.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-$root/.uv_cache}"
# Keep interpreter hash randomization fixed so paired seed comparisons are
# reproducible across separate benchmark processes.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
# Keep numerical kernels single-threaded for deterministic reductions.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/benchmark_sp500.py "$@"
elif [ -x "$root/.venv/bin/python" ]; then
  "$root/.venv/bin/python" scripts/benchmark_sp500.py "$@"
else
  python3 scripts/benchmark_sp500.py "$@"
fi
