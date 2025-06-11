import logging
from collections import Counter
from pathlib import Path
from typing import Dict


class CountingHandler(logging.Handler):
    """Logging handler that counts emitted records."""

    def __init__(self) -> None:
        super().__init__()
        self.level_counts: Counter[str] = Counter()
        self.message_counts: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        self.level_counts[record.levelname] += 1
        self.message_counts[record.getMessage()] += 1

    def get_counts(self) -> Dict[str, Dict[str, int]]:
        """Return collected counts for levels and messages."""
        return {
            "levels": dict(self.level_counts),
            "messages": dict(self.message_counts),
        }

    def dump_summary(self, path: str | Path) -> None:
        """Write a simple summary of counts to ``path``."""
        counts = self.get_counts()
        p = Path(path)
        with p.open("w", encoding="utf8") as fh:
            fh.write("[Levels]\n")
            for lvl, c in sorted(counts["levels"].items()):
                fh.write(f"{lvl}: {c}\n")
            fh.write("\n[Messages]\n")
            for msg, c in sorted(counts["messages"].items()):
                fh.write(f"{msg}: {c}\n")


_counter: CountingHandler | None = None


def enable_log_counter() -> CountingHandler:
    """Create or return the global :class:`CountingHandler` instance."""
    global _counter
    if _counter is None:
        _counter = CountingHandler()
    return _counter


def get_log_counts() -> Dict[str, Dict[str, int]]:
    """Return the counts collected by the global counter, if any."""
    if _counter is None:
        return {"levels": {}, "messages": {}}
    return _counter.get_counts()


def dump_log_counts(path: str | Path) -> None:
    """Dump counts from the global counter to ``path`` if enabled."""
    if _counter is not None:
        _counter.dump_summary(path)
