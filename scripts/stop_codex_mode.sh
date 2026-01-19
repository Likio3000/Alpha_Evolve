#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/logs/codex_mode/codex_watch.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found at $PID_FILE"
  exit 1
fi

PID="$(cat "$PID_FILE")"
if kill "$PID" >/dev/null 2>&1; then
  echo "Stopped Codex watch (PID $PID)."
  rm -f "$PID_FILE"
else
  echo "Failed to stop PID $PID (may already be stopped)."
  exit 1
fi
