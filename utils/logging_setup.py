import logging
from typing import Optional

from .log_counter import CountingHandler

def setup_logging(level: int = logging.INFO,
                  log_file: Optional[str] = None,
                  log_counter: bool | CountingHandler = False) -> CountingHandler | None:
    """Configure root logger with timestamped messages.

    Parameters
    ----------
    level : int
        Logging verbosity level.
    log_file : Optional[str]
        If given, log messages are also written to this file.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    counter_handler: CountingHandler | None = None
    if log_counter:
        counter_handler = (CountingHandler() if isinstance(log_counter, bool)
                           else log_counter)
        handlers.append(counter_handler)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

    return counter_handler

