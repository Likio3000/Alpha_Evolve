#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
touch logs/errors.log

tail -n 0 -F logs/backend.log logs/frontend.log 2>/dev/null \
  | stdbuf -oL -eL rg --line-buffered -n '(ERROR|Error:|Traceback|exception|Unhandled|EADDRINUSE|ECONN|ENOENT|ModuleNotFoundError|TypeError|ReferenceError)' \
  | awk '{ print strftime("%F %T"), $0 }' >> logs/errors.log

