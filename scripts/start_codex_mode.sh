#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AE_PYTHON:-$ROOT/.venv/bin/python}"
LOG_DIR="$ROOT/logs/codex_mode"
LOG_FILE="$LOG_DIR/codex_watch.log"
PID_FILE="$LOG_DIR/codex_watch.pid"

mkdir -p "$LOG_DIR"

if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3)"
fi

nohup "$PYTHON" "$ROOT/scripts/codex_watch.py" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Codex watch started (PID $(cat "$PID_FILE")). Log: $LOG_FILE"
