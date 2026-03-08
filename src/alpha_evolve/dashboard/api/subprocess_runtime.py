from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable


def start_text_subprocess(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )


def pump_text_subprocess_output(
    *,
    proc: subprocess.Popen[str],
    handle_line: Callable[[str], None],
    wait_timeout: float | None = None,
    handle_error: Callable[[Exception], None] | None = None,
) -> int:
    try:
        stdout = proc.stdout
        if stdout is None:
            raise RuntimeError("subprocess runner missing stdout pipe")
        for raw_line in stdout:
            handle_line(raw_line)
    except Exception as exc:
        if handle_error is not None:
            handle_error(exc)
    finally:
        try:
            if wait_timeout is None:
                code = proc.wait()
            else:
                code = proc.wait(timeout=wait_timeout)
        except Exception:
            try:
                code = proc.poll()
            except Exception:
                code = 1
        if code is None:
            code = 1
    return int(code)
