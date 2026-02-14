#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd -P)
root="$here/.."

usage() {
  cat <<'EOF'
Run benchmark_sp500 shards in parallel by splitting seed ranges.

Usage:
  bash scripts/benchmark_sp500_parallel.sh \
    --mode quick \
    --config configs/bench_sp500_scaling_monotonic_v2.toml \
    --seeds 10:30 \
    --jobs 4 \
    --outdir artifacts/bench_parallel \
    [extra benchmark_sp500 args...]

Notes:
- All extra args are forwarded to scripts/benchmark_sp500.sh.
- Creates shard directories and logs under --outdir.
- Produces --outdir/merged_runs symlinks to all run_* folders.
EOF
}

mode="quick"
config="configs/bench_sp500_small_quick.toml"
seeds_spec="0:10"
jobs=4
outdir="artifacts/bench_parallel"
extra_args=()

while (($#)); do
  case "$1" in
    --mode)
      mode="$2"
      shift 2
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    --seeds)
      seeds_spec="$2"
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
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if ! [[ "$jobs" =~ ^[0-9]+$ ]] || [[ "$jobs" -lt 1 ]]; then
  echo "--jobs must be a positive integer" >&2
  exit 1
fi

parse_seeds() {
  local spec="$1"
  local out=()
  if [[ "$spec" == *":"* ]]; then
    local start="${spec%%:*}"
    local end="${spec##*:}"
    [[ -z "$start" ]] && start=0
    if ! [[ "$start" =~ ^-?[0-9]+$ && "$end" =~ ^-?[0-9]+$ ]]; then
      echo "Invalid seed range: $spec" >&2
      exit 1
    fi
    if (( end < start )); then
      echo "Seed range must be start:end with end >= start" >&2
      exit 1
    fi
    local s
    for ((s=start; s<end; s++)); do
      out+=("$s")
    done
  elif [[ "$spec" == *,* ]]; then
    local item
    IFS=',' read -r -a raw <<< "$spec"
    for item in "${raw[@]}"; do
      item="${item//[[:space:]]/}"
      [[ -z "$item" ]] && continue
      if ! [[ "$item" =~ ^-?[0-9]+$ ]]; then
        echo "Invalid seed entry: $item" >&2
        exit 1
      fi
      out+=("$item")
    done
  else
    if ! [[ "$spec" =~ ^-?[0-9]+$ ]]; then
      echo "Invalid seed value: $spec" >&2
      exit 1
    fi
    out+=("$spec")
  fi

  if ((${#out[@]} == 0)); then
    echo "No seeds resolved from --seeds '$spec'" >&2
    exit 1
  fi

  printf '%s\n' "${out[@]}"
}

seeds=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  seeds+=("$line")
done < <(parse_seeds "$seeds_spec")

if ((${#seeds[@]} == 0)); then
  echo "No seeds to run" >&2
  exit 1
fi

if (( jobs > ${#seeds[@]} )); then
  jobs=${#seeds[@]}
fi

abs_outdir="$outdir"
if [[ "$abs_outdir" != /* ]]; then
  abs_outdir="$root/$abs_outdir"
fi
mkdir -p "$abs_outdir"

chunk_size=$(( (${#seeds[@]} + jobs - 1) / jobs ))

pids=()
shards=()

echo "[parallel-bench] mode=$mode config=$config seeds=${#seeds[@]} jobs=$jobs outdir=$abs_outdir"

for ((i=0; i<jobs; i++)); do
  start=$(( i * chunk_size ))
  (( start >= ${#seeds[@]} )) && break
  end=$(( start + chunk_size ))
  (( end > ${#seeds[@]} )) && end=${#seeds[@]}

  chunk=("${seeds[@]:start:end-start}")
  seed_csv=$(IFS=,; echo "${chunk[*]}")
  shard_id=$(printf "%02d" $((i + 1)))
  shard_dir="$abs_outdir/shard_${shard_id}"
  mkdir -p "$shard_dir"
  log="$shard_dir/run.log"

  cmd=(bash "$root/scripts/benchmark_sp500.sh"
    --mode "$mode"
    --config "$config"
    --seeds "$seed_csv"
    --outdir "$shard_dir"
  )
  if ((${#extra_args[@]} > 0)); then
    cmd+=("${extra_args[@]}")
  fi

  echo "[parallel-bench] launch shard_$shard_id seeds=$seed_csv" | tee -a "$abs_outdir/launcher.log"
  nohup "${cmd[@]}" >"$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  shards+=("$shard_id")
  echo "$pid shard_$shard_id $seed_csv" >> "$abs_outdir/pids.txt"
done

fail=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  shard="${shards[$idx]}"
  if wait "$pid"; then
    echo "[parallel-bench] shard_$shard completed" | tee -a "$abs_outdir/launcher.log"
  else
    echo "[parallel-bench] shard_$shard failed" | tee -a "$abs_outdir/launcher.log"
    fail=1
  fi
done

merged_dir="$abs_outdir/merged_runs"
rm -rf "$merged_dir"
mkdir -p "$merged_dir"

while IFS= read -r run_dir; do
  name="$(basename "$run_dir")"
  target="$merged_dir/$name"
  if [[ -e "$target" ]]; then
    stamp=$(date +%s%N)
    target="$merged_dir/${name}_$stamp"
  fi
  ln -s "$run_dir" "$target"
done < <(find "$abs_outdir" -type d -path '*/pipeline_runs_cs/run_*' | sort)

run_count=$(find "$merged_dir" -mindepth 1 -maxdepth 1 -type l | wc -l | tr -d ' ')
echo "[parallel-bench] merged run roots: $merged_dir (runs=$run_count)"

if (( fail != 0 )); then
  echo "[parallel-bench] one or more shards failed" >&2
  exit 1
fi
