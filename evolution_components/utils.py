from __future__ import annotations
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None # type: ignore

def pbar(iterable, *, desc: str, disable: bool):
    if tqdm and not disable:
        return tqdm(iterable, desc=desc, leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') # type: ignore
    class _DummyPBar:
        def __init__(self, it, **kwargs): self._it = iter(it)
        def __iter__(self): return self
        def __next__(self): return next(self._it)
        def update(self, *_args, **_kwargs): pass
        def close(self): pass
        def set_postfix_str(self, s): pass # type: ignore
    return _DummyPBar(iterable)
