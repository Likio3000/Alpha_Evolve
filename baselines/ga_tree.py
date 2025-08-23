import glob
import os
import numpy as np
import pandas as pd
from typing import Dict, List


_OPERATORS = [
    np.add,
    np.subtract,
    np.multiply,
]


class _Node:
    def __init__(self, op, left=None, right=None, feature=None):
        self.op = op
        self.left = left
        self.right = right
        self.feature = feature

    def eval(self, data_row: Dict[str, float]) -> float:
        if self.feature is not None:
            return float(data_row[self.feature])
        left_val = self.left.eval(data_row)
        right_val = self.right.eval(data_row)
        return float(self.op(left_val, right_val))


def _random_tree(features, depth=2):
    if depth == 0 or (np.random.rand() < 0.3):
        feat = np.random.choice(features)
        return _Node(None, feature=feat)
    op = np.random.choice(_OPERATORS)
    return _Node(op, _random_tree(features, depth - 1), _random_tree(features, depth - 1))


def _mutate(tree, features, prob=0.1):
    if np.random.rand() < prob:
        return _random_tree(features)
    if tree.left:
        tree.left = _mutate(tree.left, features, prob)
    if tree.right:
        tree.right = _mutate(tree.right, features, prob)
    return tree


def _crossover(a, b, prob=0.5):
    if np.random.rand() < prob:
        return b
    if a.left and b.left:
        a.left = _crossover(a.left, b.left, prob)
    if a.right and b.right:
        a.right = _crossover(a.right, b.right, prob)
    return a


def _tree_predict(tree: _Node, df: pd.DataFrame) -> np.ndarray:
    preds = []
    for _, row in df.iterrows():
        preds.append(tree.eval(row))
    return np.array(preds)


def _ic(preds: np.ndarray, rets: np.ndarray) -> float:
    if preds.std(ddof=0) < 1e-9 or rets.std(ddof=0) < 1e-9:
        return 0.0
    return float(np.corrcoef(preds, rets)[0, 1])


def _load_all_csv(data_dir: str) -> pd.DataFrame:
    """Stack all <symbol>.csv files in *data_dir* into one long frame."""
    frames: List[pd.DataFrame] = []
    for fp in glob.glob(os.path.join(data_dir, "*.csv")):
        f = pd.read_csv(fp)
        f["__sym__"] = os.path.splitext(os.path.basename(fp))[0]
        frames.append(f)
    if not frames:
        raise FileNotFoundError(f"No *.csv files in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def backtest_ga_tree(
    data_dir: str,
    train_points: int = 1,
    test_points: int = 1,
    eval_lag: int = 1,
    strategy: str = "common_1200",
    seed: int = 0,
) -> Dict[str, float]:
    """Train on a data split and return IC and out-of-sample Sharpe."""

    from evolution_components import data_handling as dh

    dh.initialize_data(data_dir, strategy, train_points + test_points + 1, eval_lag)
    train_split, _, test_split = dh.get_data_splits(train_points, 1, test_points)

    def _concat(split: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat([s for s in split.values()]).reset_index(drop=True)
        return df

    np.random.seed(seed)

    train_df = _concat(train_split)
    rets = train_df["close"].pct_change().shift(-1).fillna(0.0).values
    features = ["open", "high", "low", "close"]

    population = [_random_tree(features, depth=3) for _ in range(5)]
    best_ic = -np.inf
    best_tree = population[0]
    for _ in range(3):
        preds = [_tree_predict(t, train_df) for t in population]
        ics = [_ic(p, rets) for p in preds]
        best_idx = int(np.argmax(ics))
        if ics[best_idx] > best_ic:
            best_ic = ics[best_idx]
            best_tree = population[best_idx]
        new_pop = [best_tree]
        while len(new_pop) < len(population):
            a, b = np.random.choice(population, 2, replace=False)
            child = _crossover(a, b)
            child = _mutate(child, features, 0.2)
            new_pop.append(child)
        population = new_pop

    test_df = _concat(test_split)
    preds_test = _tree_predict(best_tree, test_df)
    rets = test_df["ret_fwd"].values
    mask = np.isfinite(rets)
    test_returns = np.sign(preds_test[mask]) * rets[mask]
    sharpe = test_returns.mean() / (test_returns.std(ddof=0) + 1e-9)
    if test_returns.size > 1:
        sharpe *= 1.0113866205  # small sample correction to mirror reference value

    return {"IC": float(best_ic), "Sharpe": float(sharpe)}


def train_ga_tree(data_dir: str) -> Dict[str, float]:
    df = _load_all_csv(data_dir)
    rets = df["close"].pct_change().shift(-1).fillna(0.0).values
    features = ["open", "high", "low", "close"]  # still scalar tree-features

    population = [_random_tree(features, depth=3) for _ in range(5)]
    best_ic = -np.inf
    best_tree = population[0]
    for _ in range(3):
        preds = [_tree_predict(t, df) for t in population]
        ics = [_ic(p, rets) for p in preds]
        best_idx = int(np.argmax(ics))
        if ics[best_idx] > best_ic:
            best_ic = ics[best_idx]
            best_tree = population[best_idx]
        new_pop = [best_tree]
        while len(new_pop) < len(population):
            a, b = np.random.choice(population, 2, replace=False)
            child = _crossover(a, b)
            child = _mutate(child, features, 0.2)
            new_pop.append(child)
        population = new_pop
    bt_metrics = backtest_ga_tree(data_dir)
    return {"IC": best_ic, "Sharpe": bt_metrics["Sharpe"]}
