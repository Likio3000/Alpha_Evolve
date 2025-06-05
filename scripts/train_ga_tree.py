import argparse
from baselines.ga_tree import train_ga_tree


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='tests/data/good')
    args = p.parse_args()
    metrics = train_ga_tree(args.data_dir)
    print(f"GA tree IC: {metrics['IC']:.4f} Sharpe: {metrics['Sharpe']:.4f}")


if __name__ == '__main__':
    main()
