#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd -P)
root="$here/.."

BASE_URL="${AE_DASHBOARD_URL:-http://127.0.0.1:8000}"
BASE_URL="${BASE_URL%/}"

SMOKE_CONFIG="${AE_SMOKE_CONFIG:-configs/bench_sp500_small_ci.toml}"
SMOKE_GENS="${AE_SMOKE_GENS:-1}"
SMOKE_POP="${AE_SMOKE_POP:-15}"
SMOKE_RUNNER_MODE="${AE_SMOKE_RUNNER_MODE:-subprocess}"

SKIP_UI_CAPTURE="${AE_SKIP_UI_CAPTURE:-0}"
SKIP_PIPELINE_SMOKE="${AE_SKIP_PIPELINE_SMOKE:-0}"

echo "[smoke] docker compose up -d --build" >&2
docker compose -f "$root/docker-compose.yml" up -d --build

echo "[smoke] waiting for $BASE_URL/health ..." >&2
for _ in $(seq 1 60); do
  if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "$BASE_URL/health" >/dev/null
echo "[smoke] health OK" >&2

echo "[smoke] ui bundle OK" >&2
curl -fsS "$BASE_URL/ui/" >/dev/null

echo "[smoke] api meta OK" >&2
curl -fsS "$BASE_URL/ui-meta/pipeline-params" >/dev/null
curl -fsS "$BASE_URL/ui-meta/evolution-params" >/dev/null

if [ "$SKIP_UI_CAPTURE" != "1" ]; then
  echo "[smoke] playwright UI navigation (capture:screens)" >&2
  npm --prefix "$root/dashboard-ui" run capture:screens
else
  echo "[smoke] AE_SKIP_UI_CAPTURE=1 (skipping Playwright UI navigation)" >&2
fi

if [ "$SKIP_PIPELINE_SMOKE" != "1" ]; then
  if [ ! -x "$root/.venv/bin/python" ]; then
    echo "[smoke] missing $root/.venv/bin/python (run scripts/setup_env.sh first)" >&2
    exit 1
  fi

  echo "[smoke] pipeline run + SSE + artefact download smoke" >&2
  SMOKE_BASE_URL="$BASE_URL" \
    SMOKE_CONFIG="$SMOKE_CONFIG" \
    SMOKE_GENS="$SMOKE_GENS" \
    SMOKE_POP="$SMOKE_POP" \
    SMOKE_RUNNER_MODE="$SMOKE_RUNNER_MODE" \
    "$root/.venv/bin/python" - <<'PY'
import json
import os
import time
import httpx

BASE = os.environ["SMOKE_BASE_URL"]
payload = {
    "generations": int(os.environ.get("SMOKE_GENS", "1")),
    "config": os.environ.get("SMOKE_CONFIG") or "configs/bench_sp500_small_ci.toml",
    "runner_mode": os.environ.get("SMOKE_RUNNER_MODE") or "subprocess",
    "overrides": {"pop_size": int(os.environ.get("SMOKE_POP", "15"))},
}

with httpx.Client(timeout=30.0) as client:
    r = client.post(f"{BASE}/api/pipeline/run", json=payload)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print("[smoke] job_id", job_id)

exit_code = None
final_msg = None
buffer = ""
start = time.time()

with httpx.Client(timeout=None) as client:
    with client.stream("GET", f"{BASE}/api/pipeline/events/{job_id}", headers={"Accept": "text/event-stream"}) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                event_type = None
                data = None
                for line in frame.splitlines():
                    if line.startswith("event:"):
                        event_type = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data = line.split(":", 1)[1].strip()
                if event_type == "ping":
                    continue
                if not data:
                    continue
                msg = json.loads(data)
                if msg.get("type") == "final":
                    final_msg = msg
                if msg.get("type") == "status" and msg.get("msg") == "exit":
                    raw_code = msg.get("code", 1)
                    try:
                        exit_code = int(raw_code)
                    except Exception:
                        exit_code = 1
                    break
            if exit_code is not None:
                break
            if time.time() - start > 300:
                raise RuntimeError("Timeout waiting for pipeline to exit (300s)")

with httpx.Client(timeout=30.0) as client:
    activity = client.get(f"{BASE}/api/job-activity/{job_id}").json()
    status = activity.get("status")
    print("[smoke] activity.status", status, "running", activity.get("running"))

if exit_code != 0:
    raise SystemExit(f"Pipeline exited with code {exit_code}")
if not final_msg or not final_msg.get("run_dir"):
    raise SystemExit("Missing final run_dir from SSE stream")

run_dir = str(final_msg["run_dir"])
print("[smoke] run_dir", run_dir)

with httpx.Client(timeout=30.0) as client:
    assets = client.get(f"{BASE}/api/run-assets", params={"run_dir": run_dir}).json().get("items", [])
    wanted = [
        "SUMMARY.json",
        "meta/ui_context.json",
        "meta/gen_summary.jsonl",
        "backtest_portfolio_csvs/return_corr_matrix.csv",
        "backtest_portfolio_csvs/return_corr_clusters.json",
        "backtest_portfolio_csvs/ensemble_selection.json",
    ]
    missing = [w for w in wanted if w not in assets]
    if missing:
        raise SystemExit(f"Missing expected artefacts: {missing}")
    for w in wanted:
        rr = client.get(f"{BASE}/api/run-asset", params={"run_dir": run_dir, "file": w})
        rr.raise_for_status()
        print("[smoke] fetched", w, rr.headers.get("content-type"))
PY
else
  echo "[smoke] AE_SKIP_PIPELINE_SMOKE=1 (skipping pipeline smoke run)" >&2
fi

echo "[smoke] done" >&2
