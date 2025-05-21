#!/usr/bin/env python
"""
run_pipeline.py – Evolve alphas *and* immediately back-test them.
Usage:  uv run run_pipeline.py 3  --fee 0  --top 20uv run run_pipeline.py 3  --fee 0  --top 20
"""
from __future__ import annotations
import argparse
import pickle
import sys
import time
from pathlib import Path

import evolve_alphas as ae
import backtest_evolved_alphas as bt

#############################################################################
# Define base output directory at the top for clarity
BASE_OUTPUT_DIR = Path("./pipeline_runs") 
#############################################################################

def _evolve_and_save(generations: int, seed: int, keep: int, run_output_dir: Path) -> Path:
    ae.args.generations = generations
    ae.args.seed = seed
    ae.args.quiet = False
    
    top = ae.evolve()
    top = top[:keep]
    
    pickle_dir = run_output_dir / "pickles"
    pickle_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    # Keep timestamp in filename for uniqueness within the pickle_dir if multiple evolutions happen
    # Or simplify if only one pkl per run_output_dir is expected
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_filename = f"evolved_{keep}x_{generations}g_{stamp}.pkl"
    pickle_filepath = pickle_dir / out_filename
    
    with open(pickle_filepath, "wb") as fh:
        pickle.dump(top, fh)
    print(f"\nSaved {keep} programs ➜ {pickle_filepath}\n")
    return pickle_filepath

#############################################################################
def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve & back-test in one go")
    ap.add_argument("generations", type=int, help="# evolutionary generations")
    ap.add_argument("--top", type=int, default=20, help="# best programs to keep")
    ap.add_argument("--fee", type=float, default=0.0, help="commission in bps")
    ap.add_argument("--hold", type=int, default=1)
    ap.add_argument("--scale", default="zscore", choices=["zscore","rank","sign"])
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--data", default=ae.DATA_DIR)
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for evolution")
    args = ap.parse_args()

    # Create a unique directory for this run
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = BASE_OUTPUT_DIR / f"run_{run_timestamp}"
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pipeline outputs will be saved in: {current_run_output_dir}")

    pkl_filepath = _evolve_and_save(args.generations, seed=args.seed, keep=args.top, run_output_dir=current_run_output_dir)
    
    # Define where backtest CSVs should go for this run
    backtest_csv_outdir = current_run_output_dir / "backtest_csvs"
    # backtest_csv_outdir.mkdir(parents=True, exist_ok=True) # bt.main will create it

    original_sys_argv = sys.argv[:] 

    sys.argv = ["backtest_evolved_alphas.py",
                "--input", str(pkl_filepath),      # Pass the specific pkl file generated
                "--top", str(args.top),
                "--fee", str(args.fee),
                "--hold", str(args.hold),
                "--scale", args.scale,
                "--lag", str(args.lag),
                "--data", str(args.data),
                "--outdir", str(backtest_csv_outdir)] # Pass the specific output dir for CSVs
    
    print(f"--- Running Backtest with args: {' '.join(sys.argv)} ---")

    bt.main()                   

    sys.argv = original_sys_argv
    print(f"--- Pipeline run finished. Check outputs in: {current_run_output_dir} ---")

#############################################################################
if __name__ == "__main__":
    main()