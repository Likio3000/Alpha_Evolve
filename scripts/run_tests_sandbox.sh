#!/usr/bin/env bash
set -euo pipefail

if [ ! -x ".venv/bin/pytest" ]; then
  echo "pytest executable not found at .venv/bin/pytest" >&2
  exit 1
fi

export SKIP_MP_TESTS="${SKIP_MP_TESTS:-1}"

PYTEST_TIMEOUT="${PYTEST_TIMEOUT:-10}"
PYTEST_CMD=( ".venv/bin/pytest" "$@" )

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
