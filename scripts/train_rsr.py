import argparse
from baselines.rsr import train_rsr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='tests/data/good')
    args = p.parse_args()
    metrics = train_rsr(args.data_dir)
    print(f"RSR IC: {metrics['IC']:.4f} Sharpe: {metrics['Sharpe']:.4f}")


if __name__ == '__main__':
    main()
