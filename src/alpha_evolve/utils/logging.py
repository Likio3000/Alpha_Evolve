from __future__ import annotations

import logging
import os
import sys
import json
import time
from typing import Dict, Optional

try:
    # Use tqdm's write to avoid breaking progress bars
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None  # type: ignore


_CONSOLE_HANDLER: "_TqdmCompatibleHandler | None" = None


class _TqdmCompatibleHandler(logging.StreamHandler):
    """A logging handler that plays nicely with tqdm progress bars.

    It routes log lines through ``tqdm.write`` when tqdm is available,
    preventing progress bars from being overwritten.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = self.format(record)
            target = (
                self.stream
                if hasattr(self, "stream") and self.stream is not None
                else sys.stdout
            )
            if _tqdm is not None:
                _tqdm.write(msg, file=target)
            else:
                target.write(msg + os.linesep)
                target.flush()
        except Exception:  # pragma: no cover
            self.handleError(record)


class _ColorFormatter(logging.Formatter):
    """ANSI colour formatter with a calm, structured aesthetic."""

    RESET = "\033[0m"
    DIM = "\033[2m"
    FG = {
        logging.CRITICAL: "\033[35m",
        logging.ERROR: "\033[31m",
        logging.WARNING: "\033[33m",
        logging.INFO: "\033[36m",
        logging.DEBUG: "\033[90m",
    }

    def __init__(self, fmt: str, datefmt: str | None, use_color: bool) -> None:
        super().__init__(fmt, datefmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        text = super().format(record)
        if not self._use_color:
            return text

        # Expect blocks separated by the unicode box-drawing divider
        try:
            ts, level, logger_name, message = text.split(" │ ", 3)
        except ValueError:
            return text

        level_colour = self.FG.get(record.levelno, self.FG[logging.DEBUG])
        ts_coloured = f"{self.DIM}{ts}{self.RESET}"
        lvl_coloured = f"{level_colour}{level}{self.RESET}"
        logger_coloured = f"{self.DIM}{logger_name}{self.RESET}"
        msg_colour = (
            level_colour if record.levelno >= logging.INFO else self.FG[logging.DEBUG]
        )
        message_coloured = f"{msg_colour}{message}{self.RESET}"

        return " │ ".join(
            (ts_coloured, lvl_coloured, logger_coloured, message_coloured)
        )


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter (Prometheus/ELK-friendly)."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: D401,N802
        fmt = datefmt or self.datefmt or "%Y-%m-%dT%H:%M:%SZ"
        return time.strftime(fmt, time.gmtime(record.created))

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        for key in (
            "client_addr",
            "request_line",
            "status_code",
            "method",
            "path",
            "query_string",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _wants_json_logs() -> bool:
    val = os.environ.get("AE_LOG_FORMAT") or os.environ.get("LOG_FORMAT") or ""
    return val.strip().lower() == "json"


def _make_console_handler(level: int) -> logging.Handler:
    """Build a tqdm-friendly, colorized console handler."""
    global _CONSOLE_HANDLER
    if _CONSOLE_HANDLER is None:
        _CONSOLE_HANDLER = _TqdmCompatibleHandler(stream=sys.stdout)
    handler = _CONSOLE_HANDLER
    handler.setLevel(level)

    # Detect whether to color: TTY and not NO_COLOR, or FORCE_COLOR set
    force_color = os.environ.get("FORCE_COLOR")
    no_color = os.environ.get("NO_COLOR") is not None
    is_tty = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
    use_color = bool(force_color or (is_tty and not no_color))

    if _wants_json_logs():
        handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ"))
    else:
        fmt = "%(asctime)s │ %(levelname)-5s │ %(name)s │ %(message)s"
        handler.setFormatter(
            _ColorFormatter(fmt, datefmt="%H:%M:%S", use_color=use_color)
        )
    return handler


def _build_uvicorn_log_config(level: int) -> Dict[str, object]:
    level_name = logging.getLevelName(level)
    if _wants_json_logs():
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "alpha_evolve.utils.logging.JsonFormatter",
                    "datefmt": "%Y-%m-%dT%H:%M:%SZ",
                },
                "access": {
                    "()": "alpha_evolve.utils.logging.JsonFormatter",
                    "datefmt": "%Y-%m-%dT%H:%M:%SZ",
                },
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "access": {
                    "class": "logging.StreamHandler",
                    "formatter": "access",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {
                    "handlers": ["default"],
                    "level": level_name,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["default"],
                    "level": level_name,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["access"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    base_fmt = "%(asctime)s │ %(levelname)-5s │ %(name)s │ %(message)s"
    access_fmt = (
        "%(asctime)s │ ACCESS │ %(client_addr)s → %(request_line)s (%(status_code)s)"
    )
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": base_fmt, "datefmt": "%H:%M:%S"},
            "access": {"format": access_fmt, "datefmt": "%H:%M:%S"},
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "class": "logging.StreamHandler",
                "formatter": "access",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": level_name,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": level_name,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }


def setup_logging(
    level: int = logging.INFO, log_file: Optional[str] = None
) -> Dict[str, object]:
    """Configure root logging and return a uvicorn-compatible log config."""
    handlers: list[logging.Handler] = [_make_console_handler(level)]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        if _wants_json_logs():
            file_handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ"))
        else:
            # File: keep date and a tidier format (seconds precision)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Silence extremely chatty third-party loggers at DEBUG level
    try:
        # Matplotlib font manager can spam DEBUG logs; keep it quiet
        import matplotlib  # type: ignore

        try:
            matplotlib.set_loglevel("warning")  # available on recent versions
        except Exception:
            pass
    except Exception:
        pass
    for noisy in (
        "matplotlib",
        "matplotlib.font_manager",
        "matplotlib.category",
    ):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass

    # Print the date once at the start to avoid per-line date clutter
    try:
        if not _wants_json_logs():
            logging.getLogger("alpha_evolve.logger").info(
                "Session started · %s", time.strftime("%Y-%m-%d")
            )
    except Exception:
        pass

    return _build_uvicorn_log_config(level)
