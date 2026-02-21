#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd -P)
root="$here/.."

usage() {
  cat <<'EOF'
Run long checkpoint scaling campaigns across one or more regimes.

Usage:
  bash scripts/run_scaling_regime_campaign.sh \
    --regime small:configs/bench_sp500_scaling_monotonic_v4.toml \
    --regime full:configs/bench_sp500_scaling_monotonic_v4_full.toml \
    --seeds 0:30 \
    --jobs 6 \
    --outdir artifacts/scaling_regimes \
    --generations 200 \
    --checkpoints 30,60,90,120,200

By default regimes are run sequentially to avoid CPU oversubscription.
Use --parallel-regimes to launch all regimes concurrently.
EOF
}

seeds="0:30"
jobs=6
outdir="artifacts/scaling_regimes"
generations=200
checkpoints="30,60,90,120,200"
parallel_regimes=0
mode="full"
skip_plots=1
regimes=()

while (($#)); do
  case "$1" in
    --regime)
      regimes+=("$2")
      shift 2
      ;;
    --seeds)
      seeds="$2"
      shift 2
      ;;
    --jobs)
      jobs="$2"
      shift 2
      ;;
    --outdir)
      outdir="$2"
      shift 2
      ;;
    --generations)
      generations="$2"
      shift 2
      ;;
    --checkpoints)
      checkpoints="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --parallel-regimes)
      parallel_regimes=1
      shift
      ;;
    --no-skip-plots)
      skip_plots=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ((${#regimes[@]} == 0)); then
  regimes+=("small:configs/bench_sp500_scaling_monotonic_v4.toml")
  regimes+=("full:configs/bench_sp500_scaling_monotonic_v4_full.toml")
fi

mkdir -p "$outdir"

run_one() {
  local label="$1"
  local config="$2"
  local regime_out="$outdir/$label"
  mkdir -p "$regime_out"
  local cmd=(bash "$root/scripts/benchmark_sp500_parallel.sh"
    --mode "$mode"
    --config "$config"
    --seeds "$seeds"
    --jobs "$jobs"
    --outdir "$regime_out"
    -- --generations "$generations" --checkpoint-gens "$checkpoints"
  )
  if (( skip_plots == 1 )); then
    cmd+=(--skip-plots)
  fi
  echo "[regime-campaign] launch $label config=$config out=$regime_out"
  "${cmd[@]}"
  echo "[regime-campaign] done $label"
}

if (( parallel_regimes == 1 )); then
  pids=()
  labels=()
  for entry in "${regimes[@]}"; do
    if [[ "$entry" != *:* ]]; then
      echo "Invalid --regime '$entry' (expected label:path)" >&2
      exit 1
    fi
    label="${entry%%:*}"
    config="${entry#*:}"
    run_one "$label" "$config" &
    pids+=("$!")
    labels+=("$label")
  done
  fail=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[regime-campaign] ${labels[$i]} completed"
    else
      echo "[regime-campaign] ${labels[$i]} failed" >&2
      fail=1
    fi
  done
  exit "$fail"
fi

for entry in "${regimes[@]}"; do
  if [[ "$entry" != *:* ]]; then
    echo "Invalid --regime '$entry' (expected label:path)" >&2
    exit 1
  fi
  label="${entry%%:*}"
  config="${entry#*:}"
  run_one "$label" "$config"
done
