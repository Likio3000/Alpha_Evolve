from __future__ import annotations

"""evolve_alphas.py  ·  v4.2 – progress bar + graceful Ctrl‑C
======================================================================
Evolutionary search for weakly‑correlated alphas **à la AlphaEvolve** on a
folder of 4‑hour OHLC crypto CSVs.

Changes vs v4.1 ──────────────────────────────────────────────────────────
* **tqdm progress‑bar** while evaluating every generation.
* `--quiet / -q` flag to disable the bar.
* Catch **Ctrl‑C** and finish the current generation so you still get the
  leaderboard instead of a messy traceback.

Usage ────────────────────────────────────────────────────────────────────
    uv run evolve_alphas.py [generations] [seed] [-q]

"""

from pathlib import Path
import argparse
import os
import glob
import math
import random
import sys
import textwrap
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # light fallback to keep zero‑dep
    tqdm = None  # type: ignore

try:
    from backtest_evolved_alphas import save_programs # Corrected name
except ImportError:
    save_programs = None
from alpha_program_core import AlphaProgram

###############################################################################
# CLI & CONFIG ################################################################
###############################################################################

def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evolutionary alpha search (crypto)")
    ap.add_argument("generations", nargs="?", type=int, default=20)
    ap.add_argument("seed", nargs="?", type=int, default=42)
    ap.add_argument("-q", "--quiet", action="store_true", help="hide progress bar")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_cli()
else:
    class _DefaultArgs:
        generations = 20
        seed = 42
        quiet = True
    args = _DefaultArgs()

DATA_DIR = "./data"
POP_SIZE = 128
N_GENERATIONS = args.generations
TOURNAMENT_K = 5
P_MUT = 0.35
P_CROSS = 0.5
ELITE_KEEP = 8
MAX_OPS = 48
PAR_SIM_PENALTY = 0.15
CORR_PENALTY_W = 0.10
CORR_CUTOFF = 0.15
DUPLICATE_HOF_SZ = 50
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)

###############################################################################
# 1. LOAD OHLC CSVs ###########################################################
###############################################################################

_FEATURE_COLS = (
    ["close", "open", "high", "low", "range"]
    + [f"ma_{w}" for w in (5, 10, 20, 30)]
    + [f"vol_{w}" for w in (5, 10, 20, 30)]
)


def _rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in (5, 10, 20, 30):
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"vol_{w}"] = df["close"].rolling(w).std(ddof=0)
    df["range"] = df["high"] - df["low"]
    df["ret_fwd"] = df["close"].pct_change().shift(-1)
    return df.dropna().reset_index(drop=True)


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if np.issubdtype(df["time"].dtype, np.integer):
        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    return _rolling_features(df)


_COINS: Dict[str, pd.DataFrame] = {
    Path(f).stem: _load_csv(f) for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
}

if not _COINS:
    sys.exit(f"No CSV files found under {DATA_DIR} – abort.")

print("Loaded", len(_COINS), "coins →", ", ".join(_COINS.keys()))

###############################################################################
# 2. SAFE CORRELATION + CACHES ################################################
###############################################################################

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson‑ρ or np.nan if either vector is constant."""
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


_HOF_expr: List[str] = []
_HOF_sigs: List[np.ndarray] = []
_eval_cache: Dict[str, Tuple[float, float]] = {}


def _concat_signal(sig_by_coin: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(list(sig_by_coin.values()))

###############################################################################
# 3. PROGRAM EXECUTION ########################################################
###############################################################################

def _run_program_on_df(prog: AlphaProgram, df: pd.DataFrame) -> np.ndarray:
    state = prog.new_state()
    sig = np.empty(len(df) - 1, dtype=float)
    for i, row in enumerate(df.itertuples(index=False)):
        features = {col: getattr(row, col) for col in _FEATURE_COLS}
        s1 = prog.eval(features, state)
        if i < len(sig):
            sig[i] = s1
    return sig


def evaluate(prog: AlphaProgram) -> Tuple[float, float]:
    fp = prog.fingerprint
    if fp in _eval_cache:
        return _eval_cache[fp]

    sigs: Dict[str, np.ndarray] = {}
    ic_vals: List[float] = []

    for sym, df in _COINS.items():
        try:
            s = _run_program_on_df(prog, df)
            sigs[sym] = s
            ic = _safe_corr(s, df["ret_fwd"].values[:-1])
            ic_vals.append(0.0 if np.isnan(ic) else ic)
        except Exception:
            n = len(df) - 1
            sigs[sym] = np.zeros(n)
            ic_vals.append(0.0)

    mean_ic = float(np.mean(ic_vals))
    score = mean_ic - PAR_SIM_PENALTY / max(1, prog.size)

    if _HOF_sigs:
        cand = _concat_signal(sigs)
        corrs = [abs(_safe_corr(cand, ref)) for ref in _HOF_sigs]
        high = [c for c in corrs if not np.isnan(c) and c > CORR_CUTOFF]
        if high:
            score -= CORR_PENALTY_W * float(np.mean(high))

    _eval_cache[fp] = (score, mean_ic)
    return _eval_cache[fp]

###############################################################################
# 4. EA OPERATORS #############################################################
###############################################################################

def _random_prog() -> AlphaProgram:
    return AlphaProgram.random_program(max_ops=MAX_OPS)


def _mutate_prog(p: AlphaProgram) -> AlphaProgram:
    return p.mutate(prob=0.1, max_ops=MAX_OPS)


def _crossover_prog(a: AlphaProgram, b: AlphaProgram) -> AlphaProgram:
    return a.crossover(b)

###############################################################################
# 5. PROGRESS BAR WRAPPER #####################################################
###############################################################################

def _pbar(iterable, *, desc: str, disable: bool):
    if tqdm and not disable:
        return tqdm(iterable, desc=desc, leave=False)
    # simple fallback
    class _Dummy:
        def __init__(self, it):
            self._it = iter(it)
        def __iter__(self):
            return self
        def __next__(self):
            return next(self._it)
        def update(self, *_):
            pass
        def close(self):
            pass
    return _Dummy(iterable)

###############################################################################
# 6. EVOLVE LOOP ##############################################################
###############################################################################

def evolve() -> List[Tuple[AlphaProgram, float]]:
    pop: List[AlphaProgram] = [_random_prog() for _ in range(POP_SIZE)]

    try:
        for gen in range(args.generations):
            t0 = time.perf_counter()
            fits: List[float] = []
            bar = _pbar(range(POP_SIZE), desc=f"Gen {gen}", disable=args.quiet)
            for i in bar:
                fit, _ = evaluate(pop[i])
                fits.append(fit)
            if tqdm and not args.quiet:
                bar.close()
            gen_time = time.perf_counter() - t0
            best_idx = int(np.argmax(fits))
            best = pop[best_idx]
            best_fit, best_ic = evaluate(best)
            print(
                f"Gen {gen:3d} | fit {best_fit:+.4f} | IC {best_ic:+.4f} | ops {best.size:3d} | "
                + textwrap.shorten(best.to_string(), width=100)
            )

            # Hall‑of‑Fame
            fp = best.fingerprint
            if fp not in _HOF_expr:
                _HOF_expr.append(fp)
                _HOF_sigs.append(_concat_signal({sym: _run_program_on_df(best, df) for sym, df in _COINS.items()}))
                if len(_HOF_expr) > DUPLICATE_HOF_SZ:
                    _HOF_expr.pop(0)
                    _HOF_sigs.pop(0)

            # elitism
            elite_idx = np.argsort(fits)[-ELITE_KEEP:][::-1]
            new_pop: List[AlphaProgram] = [pop[i].copy() for i in elite_idx]

            # breeding
            while len(new_pop) < POP_SIZE:
                a_idx = max(random.sample(range(POP_SIZE), TOURNAMENT_K), key=lambda i: fits[i])
                b_idx = max(random.sample(range(POP_SIZE), TOURNAMENT_K), key=lambda i: fits[i])
                parent_a, parent_b = pop[a_idx], pop[b_idx]
                child = _crossover_prog(parent_a, parent_b) if random.random() < P_CROSS else parent_a.copy()
                if random.random() < P_MUT:
                    child = _mutate_prog(child)
                new_pop.append(child)
            pop = new_pop

    except KeyboardInterrupt:
        print("\n[Ctrl‑C] stopping early – finishing current generation...\n")

    # final dedup top‑20
    scored = [(p, evaluate(p)[0]) for p in pop]
    scored.sort(key=lambda t: t[1], reverse=True)
    seen: set[str] = set()
    top: List[Tuple[AlphaProgram, float]] = []
    for prog, sc in scored:
        if prog.fingerprint not in seen:
            seen.add(prog.fingerprint)
            top.append((prog, sc))
        if len(top) == 20:
            break
    return top

###############################################################################
# 7. ENTRY‑POINT ##############################################################
###############################################################################

if __name__ == "__main__":
    top20 = evolve()
    if save_programs:
        save_programs(top20)  

    print("\n================ UNIQUE TOP 20 ================")
    for i, (prog, fit) in enumerate(top20, 1):
        ic = evaluate(prog)[1]
        print(
            f"#{i:02d} | fit {fit:+.4f} | IC {ic:+.4f} | ops {prog.size:3d}\n   "
            + textwrap.shorten(prog.to_string(), width=140)
            + "\n"
        )
