#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "server"
PID_FILE = LOG_DIR / "server.pid"
DEFAULT_LOG_FILE = LOG_DIR / "latest.log"
DEFAULT_WATCH_DIRS = [
    ROOT / "scripts" / "dashboard_server",
    ROOT / "scripts" / "run_dashboard.py",
    ROOT / "dashboard-ui" / "dist",
    ROOT / "evolution_components",
    ROOT / "configs",
]


def ensure_directories() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path


def parse_env_assignments(items: Sequence[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid env assignment '{raw}'. Expected KEY=VALUE.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid env assignment '{raw}'. KEY cannot be empty.")
        env[key] = value
    return env


def read_pid_info() -> Optional[Dict[str, object]]:
    if not PID_FILE.exists():
        return None
    try:
        data = json.loads(PID_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "pid" in data:
            return data
    except Exception:
        pass
    return None


def write_pid_info(info: Dict[str, object]) -> None:
    ensure_directories()
    PID_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")


def remove_pid_file() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def is_process_running(pid: int) -> bool:
    try:
        # Sending signal 0 only checks for existence.
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we may not have permission; treat as running.
        return True
    else:
        return True


def terminate_pid(pid: int, timeout: float = 10.0) -> bool:
    if not is_process_running(pid):
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Fall back to SIGKILL (if available) when lacking permission for SIGTERM.
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_process_running(pid):
            return True
        time.sleep(0.2)

    if hasattr(signal, "SIGKILL"):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True

    # Final check after attempting SIGKILL or re-issuing SIGTERM.
    time.sleep(0.5)
    return not is_process_running(pid)


def build_environment(host: Optional[str], port: Optional[int], extra: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    if host:
        env["HOST"] = host
    if port:
        env["PORT"] = str(port)
    for key, value in extra.items():
        env[key] = value
    return env


def launch_dashboard_process(env: Dict[str, str], log_path: Path) -> subprocess.Popen[bytes]:
    ensure_directories()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "scripts.run_dashboard"]
    log_file = log_path.open("ab", buffering=0)
    try:
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=env,
            close_fds=os.name != "nt",
        )
    finally:
        log_file.close()

    write_pid_info(
        {
            "pid": proc.pid,
            "command": command,
            "log_file": str(log_path),
            "started_at": time.time(),
            "env_overrides": {k: env[k] for k in ("HOST", "PORT") if k in env},
        }
    )
    return proc


def start_once(args: argparse.Namespace) -> None:
    existing = read_pid_info()
    if existing:
        pid = int(existing.get("pid", -1))
        if pid > 0 and is_process_running(pid):
            raise SystemExit(f"Server already running (pid {pid}). Use 'restart' or 'stop' first.")
        remove_pid_file()

    log_path = resolve_path(args.log_file)
    env_overrides = parse_env_assignments(args.env or [])
    env = build_environment(args.host, args.port, env_overrides)

    proc = launch_dashboard_process(env, log_path)
    print(f"Server started (pid {proc.pid}). Logging to {log_path}")


def stop_server(_: argparse.Namespace) -> None:
    info = read_pid_info()
    if not info:
        print("No PID file found. Nothing to stop.")
        return

    pid = int(info.get("pid", -1))
    if pid <= 0:
        print("Malformed PID file; removing.")
        remove_pid_file()
        return

    if not is_process_running(pid):
        print(f"Process {pid} not running. Cleaning up PID file.")
        remove_pid_file()
        return

    print(f"Stopping server process {pid}…", end="", flush=True)
    success = terminate_pid(pid)
    if success:
        print(" done.")
        remove_pid_file()
    else:
        print(" failed to terminate. Check manually.")


def restart_server(args: argparse.Namespace) -> None:
    stop_server(args)
    time.sleep(0.5)
    start_once(args)


def print_log_tail(log_path: Path, lines: int = 20) -> None:
    if not log_path.exists():
        print(f"Log file {log_path} does not exist yet.")
        return

    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        dq = deque(fh, maxlen=max(1, lines))
    for line in dq:
        print(line.rstrip("\n"))


def status_server(_: argparse.Namespace) -> None:
    info = read_pid_info()
    if not info:
        print("Server is not running (no PID file).")
        return

    pid = int(info.get("pid", -1))
    running = pid > 0 and is_process_running(pid)
    log_path = Path(info.get("log_file", DEFAULT_LOG_FILE))
    if not log_path.is_absolute():
        log_path = ROOT / log_path

    if running:
        print(f"Server running (pid {pid}). Log: {log_path}")
    else:
        print(f"PID file present but process {pid} not running. Log: {log_path}")
        remove_pid_file()

    print("--- Log tail ---")
    print_log_tail(log_path)


def tail_log(args: argparse.Namespace) -> None:
    log_path = resolve_path(args.log_file)
    if not log_path.exists():
        print(f"{log_path} does not exist yet.")
        if not args.follow:
            return
        print("Waiting for log file to appear…")
        while not log_path.exists():
            time.sleep(0.5)

    def print_initial_lines() -> None:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            if args.lines > 0:
                for line in deque(fh, maxlen=args.lines):
                    print(line.rstrip("\n"))
            else:
                for line in fh:
                    print(line.rstrip("\n"))

    print_initial_lines()
    if not args.follow:
        return

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, os.SEEK_END)
            while True:
                line = fh.readline()
                if not line:
                    time.sleep(0.25)
                    continue
                print(line, end="")
    except KeyboardInterrupt:
        pass


def compute_watch_paths(args: argparse.Namespace) -> Tuple[List[Path], List[Path]]:
    paths: List[Path] = []
    if args.no_default_watch:
        pass
    else:
        paths.extend(DEFAULT_WATCH_DIRS)
    for raw in args.watch_path or []:
        paths.append(resolve_path(raw))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    existing: List[Path] = []
    missing: List[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            existing.append(path)
        else:
            missing.append(path)
    return existing, missing


def watch_and_run(args: argparse.Namespace) -> None:
    try:
        from watchfiles import watch  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "watchfiles is required for --watch mode. Install it with 'pip install watchfiles'."
        ) from exc

    watch_paths, missing_paths = compute_watch_paths(args)
    if not watch_paths:
        raise SystemExit("No valid watch paths found. Specify with --watch-path.")

    print("[watch] Watching directories:")
    for path in watch_paths:
        print(f"  - {path}")
    if missing_paths:
        print("[watch] Skipping missing paths:")
        for path in missing_paths:
            print(f"  - {path}")

    env_overrides = parse_env_assignments(args.env or [])
    env = build_environment(args.host, args.port, env_overrides)
    log_path = resolve_path(args.log_file)
    stop_event = threading.Event()
    proc_holder: Dict[str, Optional[subprocess.Popen[bytes]]] = {"proc": None}

    def spawn(label: str) -> None:
        proc = launch_dashboard_process(env, log_path)
        proc_holder["proc"] = proc
        print(f"[watch] {label}: started pid {proc.pid}")

    def stop_process(reason: str) -> None:
        proc = proc_holder.get("proc")
        if not proc:
            return
        if proc.poll() is not None:
            remove_pid_file()
            return
        print(f"[watch] {reason}: stopping pid {proc.pid}")
        terminated = terminate_pid(proc.pid)
        if not terminated:
            print(f"[watch] Warning: could not terminate pid {proc.pid}")
        proc_holder["proc"] = None
        remove_pid_file()

    def watcher_thread() -> None:
        try:
            for changes in watch(*watch_paths, stop_event=stop_event, debounce=500):
                if stop_event.is_set():
                    break
                printable = ", ".join(str(p) for _, p in changes)
                print(f"[watch] Detected changes: {printable}")
                stop_process("file change")
                spawn("restart")
        finally:
            stop_event.set()

    spawn("initial start")

    watcher = threading.Thread(target=watcher_thread, daemon=True)
    watcher.start()

    try:
        while not stop_event.is_set():
            proc = proc_holder.get("proc")
            if proc and proc.poll() is not None:
                code = proc.returncode
                print(f"[watch] Process exited with code {code}. Restarting…")
                remove_pid_file()
                spawn("recovery")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[watch] Caught interrupt, shutting down…")
        stop_event.set()
    finally:
        stop_process("shutdown")
        watcher.join(timeout=2.0)


def start_server(args: argparse.Namespace) -> None:
    if args.watch:
        watch_and_run(args)
    else:
        start_once(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the Alpha Evolve dashboard server.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start the dashboard server.")
    start_parser.add_argument("--host", help="Override HOST environment variable.")
    start_parser.add_argument("--port", type=int, help="Override PORT environment variable.")
    start_parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE), help="Log file path.")
    start_parser.add_argument(
        "--env",
        action="append",
        help="Extra environment variable assignments (KEY=VALUE). May be specified multiple times.",
    )
    start_parser.add_argument("--watch", action="store_true", help="Enable watch mode with auto-restart.")
    start_parser.add_argument(
        "--watch-path",
        action="append",
        help="Additional paths to watch for changes. May be repeated.",
    )
    start_parser.add_argument(
        "--no-default-watch",
        action="store_true",
        help="When used with --watch, only watch the paths supplied via --watch-path.",
    )
    start_parser.set_defaults(func=start_server)

    stop_parser = subparsers.add_parser("stop", help="Stop the dashboard server.")
    stop_parser.set_defaults(func=stop_server)

    restart_parser = subparsers.add_parser("restart", help="Restart the dashboard server.")
    restart_parser.add_argument("--host", help="Override HOST environment variable.")
    restart_parser.add_argument("--port", type=int, help="Override PORT environment variable.")
    restart_parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE), help="Log file path.")
    restart_parser.add_argument(
        "--env",
        action="append",
        help="Extra environment variable assignments (KEY=VALUE). May be specified multiple times.",
    )
    restart_parser.set_defaults(func=restart_server)

    status_parser = subparsers.add_parser("status", help="Show server status and log tail.")
    status_parser.set_defaults(func=status_server)

    tail_parser = subparsers.add_parser("tail", help="Tail the server log.")
    tail_parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE), help="Log file path.")
    tail_parser.add_argument("--lines", type=int, default=40, help="Number of initial lines to print.")
    tail_parser.add_argument("--follow", action="store_true", help="Follow the log (like tail -f).")
    tail_parser.set_defaults(func=tail_log)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
