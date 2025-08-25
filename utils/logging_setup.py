import logging
import os
import sys
from typing import Optional

try:
    # Use tqdm's write to avoid breaking progress bars
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None  # type: ignore


class _TqdmCompatibleHandler(logging.StreamHandler):
    """A logging handler that plays nicely with tqdm progress bars.

    It routes log lines through ``tqdm.write`` when tqdm is available,
    preventing progress bars from being overwritten.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = self.format(record)
            if _tqdm is not None:
                _tqdm.write(msg)
            else:
                stream = self.stream if hasattr(self, "stream") else sys.stdout
                stream.write(msg + os.linesep)
                stream.flush()
        except Exception:  # pragma: no cover
            self.handleError(record)


class _ColorFormatter(logging.Formatter):
    """Minimal ANSI color formatter for console readability.

    Colors only when the output stream is a TTY and NO_COLOR is not set.
    """

    # ANSI escape sequences
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    FG = {
        "grey": "\033[90m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }

    def __init__(self, fmt: str, datefmt: str | None, use_color: bool) -> None:
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        base = super().format(record)
        if not self.use_color:
            return base

        level = record.levelno
        if level >= logging.ERROR:
            color = self.FG["red"]
        elif level >= logging.WARNING:
            color = self.FG["yellow"]
        elif level >= logging.INFO:
            color = self.FG["green"]
        else:
            color = self.FG["cyan"]

        # Lightly dim timestamp and location, keep message colored by level
        # Expect input format like: "2025-08-23 11:49:39,291 [INFO] module:123 | message"
        try:
            prefix, message = base.split("| ", 1)
            prefix_col = f"{self.DIM}{prefix}{self.RESET}"
            return f"{color}{message}{self.RESET}" if not prefix else f"{prefix_col} | {color}{message}{self.RESET}"
        except Exception:
            return f"{color}{base}{self.RESET}"


def _make_console_handler(level: int) -> logging.Handler:
    """Build a tqdm-friendly, colorized console handler."""
    handler = _TqdmCompatibleHandler(stream=sys.stdout)
    handler.setLevel(level)

    # Detect whether to color: TTY and not NO_COLOR, or FORCE_COLOR set
    force_color = os.environ.get("FORCE_COLOR")
    no_color = os.environ.get("NO_COLOR") is not None
    is_tty = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
    use_color = bool(force_color or (is_tty and not no_color))

    # Lighter format: time only (HH:MM:SS) and no module/lineno in console
    fmt = "%(asctime)s [%(levelname)s] | %(message)s"
    handler.setFormatter(_ColorFormatter(fmt, datefmt="%H:%M:%S", use_color=use_color))
    return handler


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure root logger with tqdm-friendly, colored console output.

    - Console: colorized, compatible with tqdm progress bars
    - File (optional): plain text without ANSI codes
    """
    handlers: list[logging.Handler] = [_make_console_handler(level)]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        # File: keep date and a tidier format (seconds precision)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Print the date once at the start to avoid per-line date clutter
    try:
        import time as _t
        logging.getLogger(__name__).info("Date %s", _t.strftime("%Y-%m-%d"))
    except Exception:
        pass
