#!/usr/bin/env python
"""
backtest_evolved_alphas.py  ·  v0.2
────────────────────────────────────────────────────────────────────────────
Back-test the top-N AlphaProgram objects saved by evolve_alphas.py.

  uv run backtest_evolved_alphas.py --top 20 --data ./data --fee 1.0
"""
from __future__ import annotations
import argparse, pickle, sys, textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── safe import – silence alpha_evolve's CLI parser ──────────────────────
_saved_argv = sys.argv[:]                 # keep user args safe
sys.argv = ["alpha_evolve_import"]        # dummy argv for the import
import evolve_alphas as ae       # ──▶ now it sees no flags
sys.argv = _saved_argv                    # restore real argv
# ------------------------------------------------------------------------

from backtest_alpha import _backtest_one  # reuse proven back-tester

PICKLE = "evolved_top20.pkl"              # written by save_programs()

# ──────────────────────────────────────────────────────────────────────────
def save_programs(progs, path="evolved_top20.pkl"):
    """Persist [(AlphaProgram, fitness), …] to a pickle."""
    import pickle
    with open(path, "wb") as fh:
        # deep-copy to detach from any open state objects
        pickle.dump([(p.copy(), fit) for p, fit in progs], fh)

def _pad_signal(sig: np.ndarray, target: int) -> np.ndarray:
    """
    AlphaProgram emits len(df)-1 observations (it can’t know tomorrow’s
    return).  We append one neutral ‘0’ so it lines up with the price series.
    """
    if len(sig) == target - 1:
        return np.append(sig, 0.0)
    if len(sig) != target:
        raise ValueError(f"signal {len(sig)} vs series {target}")
    return sig

def load_ohlc(folder: str) -> Dict[str, pd.DataFrame]:
    dfs = {}
    for csv in Path(folder).glob("*.csv"):
        df = pd.read_csv(csv)
        df = ae._rolling_features(df)
        dfs[csv.stem] = df.reset_index(drop=True)
    if not dfs:
        raise SystemExit(f"No CSV files in {folder}")
    return dfs

def load_programs(n: int, pickle_path: str | None = None) \
        -> List[Tuple[ae.AlphaProgram, float]]:
    path = pickle_path or PICKLE          # fall back to default
    with open(path, "rb") as fh:
        progs: List[Tuple[ae.AlphaProgram, float]] = pickle.load(fh)
    return progs[:n]

# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Back-test evolved alphas")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--input", default=PICKLE)
    ap.add_argument("--data", default=ae.DATA_DIR)
    ap.add_argument("--fee", type=float, default=0.0,
                    help="round-trip commission in bps")
    ap.add_argument("--hold", type=int, default=1)
    ap.add_argument("--scale", choices=["zscore", "rank", "sign"],
                    default="zscore")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--outdir", default="evolved_bt")
    args = ap.parse_args()

    Path(args.outdir).mkdir(exist_ok=True)
    dfs = load_ohlc(args.data)
    programs = load_programs(args.top, args.input)

    for idx, (prog, fit) in enumerate(programs, 1):
        rows = []
        for pair, df in dfs.items():
            sig = ae._run_program_on_df(prog, df)
            sig = _pad_signal(sig, len(df))  # <<< length-fix
            rows.append({
                "Pair": pair,
                **_backtest_one(df, sig,
                                fee_bps=args.fee,
                                lag=args.lag,
                                hold=args.hold,
                                scale=args.scale)
            })
        res = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)

        # equal-weight portfolio line
        avg = res[["Sharpe", "AnnReturn", "AnnVol", "MaxDD", "Turnover"]].mean()
        res.loc[len(res)] = ["EW-portfolio",
                     avg["Sharpe"], avg["AnnReturn"],
                     avg["AnnVol"],  avg["MaxDD"],
                     avg["Turnover"], float("nan")]

        csv_path = Path(args.outdir) / f"alpha_{idx:02d}_fit{fit:+.4f}.csv"
        res.to_csv(csv_path, index=False)
        print(f"[{idx:02d}] fit {fit:+.4f} → {csv_path}  ::  "
              f"{textwrap.shorten(prog.to_string(), 100)}")

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
