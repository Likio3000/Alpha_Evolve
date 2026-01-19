#!/usr/bin/env bash
set -euo pipefail

PYTEST_BIN=".venv/bin/pytest"
if [ ! -x "$PYTEST_BIN" ]; then
  PYTEST_BIN=".venv312/bin/pytest"
fi
if [ ! -x "$PYTEST_BIN" ]; then
  echo "pytest executable not found at .venv/bin/pytest or .venv312/bin/pytest" >&2
  exit 1
fi

export SKIP_MP_TESTS="${SKIP_MP_TESTS:-1}"

PYTEST_TIMEOUT="${PYTEST_TIMEOUT:-10}"
PYTEST_CMD=( "$PYTEST_BIN" "$@" )

if command -v timeout >/dev/null 2>&1; then
  timeout --signal=INT "${PYTEST_TIMEOUT}s" "${PYTEST_CMD[@]}"
  status=$?
  if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    echo "pytest exceeded ${PYTEST_TIMEOUT}s sandbox timeout; cancelling run." >&2
    exit 124
  fi
  exit "$status"
else
  exec "${PYTEST_CMD[@]}"
fi
