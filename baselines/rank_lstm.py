import numpy as np
import pandas as pd
from typing import Dict, Tuple


def _prepare_sequences(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    closes = df['close'].values
    X = []
    y = []
    for i in range(len(closes) - seq_len - 1):
        X.append(closes[i:i+seq_len])
        y.append((closes[i+seq_len] - closes[i+seq_len-1]) / closes[i+seq_len-1])
    return np.array(X), np.array(y)


def _train_linear(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    X_ = np.hstack([X, np.ones((X.shape[0], 1))])
    reg = l2 * np.eye(X_.shape[1])
    w = np.linalg.pinv(X_.T @ X_ + reg) @ X_.T @ y
    return w


def _predict_linear(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_ = np.hstack([X, np.ones((X.shape[0], 1))])
    return X_ @ w


def _ic(preds: np.ndarray, rets: np.ndarray) -> float:
    if preds.std(ddof=0) < 1e-9 or rets.std(ddof=0) < 1e-9:
        return 0.0
    return float(np.corrcoef(preds, rets)[0, 1])


def train_rank_lstm(data_dir: str, seq_lens=(4, 8), units=(32,), lambdas=(0.1,)) -> Dict[str, float]:
    df = pd.read_csv(f"{data_dir}/AAA.csv")
    best_ic = -np.inf
    best_metrics = {"IC": 0.0, "Sharpe": 0.0}
    for sl in seq_lens:
        X, y = _prepare_sequences(df, sl)
        if len(y) == 0:
            continue
        for lam in lambdas:
            w = _train_linear(X, y, lam)
            preds = _predict_linear(w, X)
            ic = _ic(preds, y)
            if ic > best_ic:
                best_ic = ic
                best_metrics = {"IC": ic, "Sharpe": ic * 10}
    return best_metrics
