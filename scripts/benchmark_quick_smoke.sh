#!/usr/bin/env bash
set -euo pipefail

# Sandbox/CI-friendly benchmark smoke test (single seed).
#
# Usage:
#   bash scripts/benchmark_quick_smoke.sh
# Env:
#   SEEDS="0:2"  # optional

SEEDS="${SEEDS:-0:1}"
BENCH_CONFIG="${BENCH_CONFIG:-configs/bench_sp500_small_quick.toml}"

bash scripts/benchmark_sp500.sh \
  --mode quick \
  --config "${BENCH_CONFIG}" \
  --seeds "${SEEDS}" \
  --outdir artifacts/benchmarks
