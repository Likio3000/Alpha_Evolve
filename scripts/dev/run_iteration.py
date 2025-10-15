#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import httpx


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts"
LOG_ROOT = ARTIFACT_ROOT / "logs"
SCREENSHOT_ROOT = ARTIFACT_ROOT / "screenshots"
SERVER_MANAGER = ROOT / "scripts" / "dev" / "server_manager.py"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def run_command(
    command: list[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    pretty = " ".join(command)
    print(f"[iterate] Running: {pretty}")
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )
    if capture_output and result.stdout:
        print(result.stdout.strip())
    return result


def wait_for_dashboard(url: str, timeout: float) -> None:
    print(f"[iterate] Waiting for dashboard at {url} (timeout {timeout:.1f}s)…")
    start = time.time()
    last_error: Optional[Exception] = None
    while time.time() - start < timeout:
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(url)
                if resp.status_code < 500:
                    print(f"[iterate] Dashboard responded with HTTP {resp.status_code}.")
                    return
        except Exception as exc:  # pragma: no cover - best effort wait loop
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Dashboard did not become ready within {timeout:.1f}s. Last error: {last_error}")


def start_server(host: str, port: int, log_file: Path, extra_env: Dict[str, str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SERVER_MANAGER),
        "start",
        "--host",
        host,
        "--port",
        str(port),
        "--log-file",
        str(log_file),
    ]
    for key, value in extra_env.items():
        command.extend(["--env", f"{key}={value}"])
    run_command(command)


def stop_server() -> None:
    run_command([sys.executable, str(SERVER_MANAGER), "stop"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate an iteration: start backend, build UI, capture screenshots.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Dashboard host (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Dashboard port (default: %(default)s)")
    parser.add_argument(
        "--dashboard-url",
        default=None,
        help="Full dashboard URL to ping (default: http://<host>:<port>/ui/)",
    )
    parser.add_argument(
        "--log-dir",
        default=str(LOG_ROOT),
        help="Directory for server logs (default: artifacts/logs)",
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip `npm run build` before capturing screenshots.")
    parser.add_argument("--reuse-server", action="store_true", help="Assume server is already running; do not start/stop.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds to wait for dashboard readiness.")
    parser.add_argument("--env", action="append", default=[], help="Extra env assignments for server start (KEY=VALUE).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_dir)
    log_file = log_dir / f"dashboard-{timestamp}.log"
    if args.dashboard_url:
        dashboard_url = args.dashboard_url
    else:
        dashboard_url = f"http://{args.host}:{args.port}/ui/"
    if not dashboard_url.endswith("/"):
        dashboard_url = f"{dashboard_url}/"
    server_started = False

    extra_env = {}
    for assignment in args.env:
        if "=" not in assignment:
            raise SystemExit(f"Invalid env assignment '{assignment}'. Expected KEY=VALUE.")
        key, value = assignment.split("=", 1)
        extra_env[key.strip()] = value

    try:
        if not args.reuse_server:
            start_server(args.host, args.port, log_file, extra_env)
            server_started = True

        wait_for_dashboard(dashboard_url, args.timeout)

        if not args.skip_build:
            run_command(["npm", "run", "build"], cwd=ROOT / "dashboard-ui")

        env = os.environ.copy()
        env["AE_DASHBOARD_URL"] = dashboard_url
        run_command(["npm", "run", "capture:screens"], cwd=ROOT / "dashboard-ui", env=env)

        latest_manifest = SCREENSHOT_ROOT / "latest.json"
        if latest_manifest.exists():
            data = json.loads(latest_manifest.read_text(encoding="utf-8"))
            print("[iterate] Latest screenshot manifest:")
            print(json.dumps(data, indent=2))
        else:
            print("[iterate] Warning: latest screenshot manifest not found.")
    finally:
        if server_started and not args.reuse_server:
            try:
                stop_server()
            except Exception as exc:
                print(f"[iterate] Failed to stop server cleanly: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
