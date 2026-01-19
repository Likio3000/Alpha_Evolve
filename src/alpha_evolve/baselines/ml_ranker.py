from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor

from alpha_evolve.backtesting.core import backtest_cross_sectional_alpha
from alpha_evolve.backtesting.data import load_and_align_data_for_backtest
from alpha_evolve.config import BacktestConfig
from alpha_evolve.programs.types import (
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    SCALAR_FEATURE_NAMES,
)
from alpha_evolve.utils.errors import DataLoadError


_EXCLUDED_FEATURES = {"sector_id_vector"}


def _feature_names() -> list[str]:
    return [f for f in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES if f not in _EXCLUDED_FEATURES]


def _features_to_matrix(
    features_at_t: Dict[str, np.ndarray],
    feature_names: Sequence[str],
    n_stocks: int,
) -> np.ndarray:
    cols: List[np.ndarray] = []
    for name in feature_names:
        vec = features_at_t.get(name)
        if vec is None:
            cols.append(np.zeros(n_stocks, dtype=float))
            continue
        arr = np.asarray(vec, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != n_stocks:
            cols.append(np.zeros(n_stocks, dtype=float))
            continue
        cols.append(arr)
    mat = np.column_stack(cols) if cols else np.zeros((n_stocks, 0), dtype=float)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def _zscore_cross_section(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0, ddof=0)
    std = np.where(std < 1e-9, 1.0, std)
    out = (mat - mean) / std
    return np.clip(out, -6.0, 6.0)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    if np.std(a, ddof=0) < 1e-9 or np.std(b, ddof=0) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class _MLSignalProgram:
    predictions: np.ndarray

    def new_state(self) -> Dict[str, Any]:
        return {"ml_step": 0}

    def eval(
        self,
        _features_at_t: Dict[str, Any],
        state: Dict[str, Any],
        n_stocks: int,
    ) -> np.ndarray:
        step = int(state.get("ml_step", 0))
        state["ml_step"] = step + 1
        if step < 0 or step >= self.predictions.shape[0]:
            return np.zeros(n_stocks, dtype=float)
        vec = self.predictions[step]
        if vec.shape[0] != n_stocks:
            return np.zeros(n_stocks, dtype=float)
        return vec


def _build_ret_fwd_matrix(
    aligned_dfs,
    stock_symbols: Sequence[str],
    time_index: Sequence,
) -> np.ndarray:
    cols = []
    for sym in stock_symbols:
        series = aligned_dfs[sym]["ret_fwd"].loc[time_index].values
        cols.append(np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0))
    return np.column_stack(cols)


def _compute_splits(total_points: int) -> tuple[int, int]:
    min_test = 200
    max_test = max(50, int(total_points * 0.3))
    test_points = min(max(min_test, int(total_points * 0.2)), max_test, total_points - 50)
    train_points = total_points - test_points
    return train_points, test_points


def train_ml_ranker(
    cfg: BacktestConfig,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    try:
        aligned_dfs, common_index, stock_symbols = load_and_align_data_for_backtest(
            cfg.data_dir,
            cfg.max_lookback_data_option,
            cfg.min_common_points,
            cfg.eval_lag,
        )
    except DataLoadError as exc:
        return {"IC": 0.0, "Sharpe": 0.0, "Error": str(exc)}
    n_stocks = len(stock_symbols)
    total_points = max(0, len(common_index) - int(cfg.eval_lag))
    if total_points < 100:
        return {"IC": 0.0, "Sharpe": 0.0}

    train_points, test_points = _compute_splits(total_points)
    if train_points < 50 or test_points < 20:
        return {"IC": 0.0, "Sharpe": 0.0}

    feature_names = _feature_names()

    from alpha_evolve.backtesting.core import _get_feature_bundle_cached
    from alpha_evolve.evolution.data import get_sector_groups

    sector_groups_vec = get_sector_groups(list(stock_symbols)).astype(float)
    bundle = _get_feature_bundle_cached(
        aligned_dfs,
        list(stock_symbols),
        common_index,
        SCALAR_FEATURE_NAMES,
        CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
        sector_groups_vec,
    )

    ret_fwd = _build_ret_fwd_matrix(aligned_dfs, stock_symbols, common_index)

    X_train_parts: List[np.ndarray] = []
    y_train_parts: List[np.ndarray] = []
    for t in range(train_points):
        features_at_t = bundle.features_per_time[t]
        X_t = _zscore_cross_section(
            _features_to_matrix(features_at_t, feature_names, n_stocks)
        )
        y_t = ret_fwd[t]
        mask = np.isfinite(y_t) & np.all(np.isfinite(X_t), axis=1)
        if not np.any(mask):
            continue
        X_train_parts.append(X_t[mask])
        y_train_parts.append(y_t[mask])

    if not X_train_parts:
        return {"IC": 0.0, "Sharpe": 0.0}

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)

    model = HistGradientBoostingRegressor(
        max_depth=6,
        max_iter=240,
        learning_rate=0.05,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    test_start = train_points
    test_end = train_points + test_points
    test_index = common_index[test_start:test_end]

    preds_matrix = np.zeros((test_points, n_stocks), dtype=float)
    ic_values: List[float] = []
    for offset, t in enumerate(range(test_start, test_end)):
        features_at_t = bundle.features_per_time[t]
        X_t = _zscore_cross_section(
            _features_to_matrix(features_at_t, feature_names, n_stocks)
        )
        preds = model.predict(X_t)
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        preds_matrix[offset] = preds
        corr = _safe_corr(preds, ret_fwd[t])
        if corr is not None:
            ic_values.append(corr)

    test_dfs = {sym: df.loc[test_index] for sym, df in aligned_dfs.items()}
    metrics = backtest_cross_sectional_alpha(
        prog=_MLSignalProgram(preds_matrix),
        aligned_dfs=test_dfs,
        common_time_index=test_index,
        stock_symbols=list(stock_symbols),
        n_stocks=n_stocks,
        fee_bps=cfg.fee,
        lag=cfg.eval_lag,
        hold=cfg.hold,
        scale_method=cfg.scale,
        long_short_n=cfg.long_short_n,
        net_exposure_target=cfg.net_exposure_target,
        winsor_p=cfg.winsor_p,
        debug_prints=False,
        annualization_factor=cfg.annualization_factor,
        stop_loss_pct=cfg.stop_loss_pct,
        sector_neutralize_positions=cfg.sector_neutralize_positions,
        volatility_target=cfg.volatility_target,
        volatility_lookback=cfg.volatility_lookback,
        max_leverage=cfg.max_leverage,
        min_leverage=cfg.min_leverage,
        dd_limit=cfg.dd_limit,
        dd_reduction=cfg.dd_reduction,
        initial_state_vars_config={"ml_step": "scalar"},
        scalar_feature_names=SCALAR_FEATURE_NAMES,
        cross_sectional_feature_vector_names=CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    )

    ic = float(np.mean(ic_values)) if ic_values else 0.0
    return {
        "IC": ic,
        "Sharpe": float(metrics.get("Sharpe", 0.0)),
        "AnnReturn": float(metrics.get("AnnReturn", 0.0)),
        "AnnVol": float(metrics.get("AnnVol", 0.0)),
        "MaxDD": float(metrics.get("MaxDD", 0.0)),
        "Turnover": float(metrics.get("Turnover", 0.0)),
        "TrainPoints": int(train_points),
        "TestPoints": int(test_points),
        "Model": "HistGradientBoostingRegressor",
        "FeatureCount": len(feature_names),
        "Features": feature_names,
        "Params": {
            "max_depth": 6,
            "max_iter": 240,
            "learning_rate": 0.05,
            "l2_regularization": 1.0,
        },
    }
