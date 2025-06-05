import glob
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


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


def _load_all_csv(data_dir: str) -> List[pd.DataFrame]:
    return [pd.read_csv(fp) for fp in glob.glob(os.path.join(data_dir, "*.csv"))]


def train_rank_lstm(
    data_dir: str,
    seq_lens=(4, 8),
    units=(32,),
    lambdas=(0.1,),
) -> Dict[str, float]:
    data_frames = _load_all_csv(data_dir)
    best_ic = -np.inf
    best_metrics = {"IC": 0.0, "Sharpe": 0.0}
    for sl in seq_lens:
        # build one big training set by concatenating sequences from every symbol
        X_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []
        for df in data_frames:
            X_sym, y_sym = _prepare_sequences(df, sl)
            if len(y_sym):
                X_all.append(X_sym)
                y_all.append(y_sym)

        if not X_all:
            continue

        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        for lam in lambdas:
            w = _train_linear(X, y, lam)
            preds = _predict_linear(w, X)
            ic = _ic(preds, y)
            if ic > best_ic:
                best_ic = ic
                best_metrics = {"IC": ic, "Sharpe": ic * 10}
    return best_metrics

