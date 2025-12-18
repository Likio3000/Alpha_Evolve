#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd -P)
root="$here/.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-$root/.uv_cache}"

PYTEST_TIMEOUT="${PYTEST_TIMEOUT:-60}"
export PYTEST_TIMEOUT

echo "[verify] Python tests (sandbox-friendly)..." >&2
"$root/scripts/run_tests_sandbox.sh"

echo "[verify] Dashboard UI lint + build..." >&2
npm --prefix "$root/dashboard-ui" run lint
npm --prefix "$root/dashboard-ui" run build

echo "[verify] Smoke pipeline run..." >&2
bash "$root/scripts/smoke_run.sh"

echo "[verify] All checks passed." >&2
