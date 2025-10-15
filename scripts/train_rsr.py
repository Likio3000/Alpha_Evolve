import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha_evolve.baselines.rsr import train_rsr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='tests/data/good')
    args = p.parse_args()
    metrics = train_rsr(args.data_dir)
    print(f"RSR IC: {metrics['IC']:.4f} Sharpe: {metrics['Sharpe']:.4f}")


if __name__ == '__main__':
    main()
