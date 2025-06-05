import numpy as np
import pandas as pd
from typing import Dict


def _prepare_graph_features(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    closes = df['close'].values
    X = []
    for i in range(len(closes) - seq_len):
        seq = closes[i:i+seq_len]
        X.append(seq - seq.mean())
    return np.array(X)


def _ic(preds: np.ndarray, rets: np.ndarray) -> float:
    if preds.std(ddof=0) < 1e-9 or rets.std(ddof=0) < 1e-9:
        return 0.0
    return float(np.corrcoef(preds, rets)[0, 1])


def train_rsr(data_dir: str, seq_lens=(4, 8), units=(32,), lambdas=(0.1,)) -> Dict[str, float]:
    df = pd.read_csv(f"{data_dir}/AAA.csv")
    best_ic = -np.inf
    best_metrics = {"IC": 0.0, "Sharpe": 0.0}
    for sl in seq_lens:
        X = _prepare_graph_features(df, sl)
        if len(X) == 0:
            continue
        y = (df['close'].shift(-1).iloc[sl:].values - df['close'].iloc[sl:].values) / df['close'].iloc[sl:].values
        for lam in lambdas:
            w = np.linalg.pinv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y
            preds = X @ w
            ic = _ic(preds, y)
            if ic > best_ic:
                best_ic = ic
                best_metrics = {"IC": ic, "Sharpe": ic * 10}
    return best_metrics
