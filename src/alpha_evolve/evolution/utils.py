"""Minimal progress-bar shim used throughout the project.

This module attempts to import :func:`tqdm.tqdm` to provide rich progress
bar functionality.  If :mod:`tqdm` is unavailable, a lightweight dummy
iterator with ``update`` and ``close`` methods is returned instead so that
client code does not need to handle the optional dependency.
"""

from __future__ import annotations
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None # type: ignore

def pbar(iterable, *, desc: str, disable: bool, total: int | None = None):
    """Return an iterable wrapped in a progress bar.

    Parameters
    ----------
    iterable:
        Any iterable to be progressed through.
    desc:
        Text displayed to the left of the progress bar.
    disable:
        If ``True`` or if :mod:`tqdm` is not available, the iterable is
        returned without a visible progress bar.
    total:
        Optional total number of expected iterations, forwarded to
        :func:`tqdm.tqdm` when available.

    Returns
    -------
    Iterable
        An iterator yielding the same items as ``iterable``. When ``tqdm``
        is available a progress bar is displayed; otherwise a dummy iterator
        with ``update`` and ``close`` methods is provided for compatibility.
    """
    if tqdm and not disable:
        return tqdm(iterable, desc=desc, total=total, leave=False, ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')  # type: ignore
    class _DummyPBar:
        def __init__(self, it, **_kwargs):
            self._it = iter(it)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._it)

        def update(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

        def set_postfix_str(self, _s):
            pass  # type: ignore

    return _DummyPBar(iterable)
