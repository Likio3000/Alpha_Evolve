from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class ModelPreset:
    name: str
    params: Dict[str, Any]
    description: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    description: str
    factory: Callable[[Dict[str, Any]], Any]
    presets: List[ModelPreset]
    supports_random_state: bool = False
    seed_param: str | None = None


def _linear_regression(params: Dict[str, Any]) -> Any:
    from sklearn.linear_model import LinearRegression

    return LinearRegression(**params)


def _ridge(params: Dict[str, Any]) -> Any:
    from sklearn.linear_model import Ridge

    return Ridge(**params)


def _lasso(params: Dict[str, Any]) -> Any:
    from sklearn.linear_model import Lasso

    return Lasso(**params)


def _elastic_net(params: Dict[str, Any]) -> Any:
    from sklearn.linear_model import ElasticNet

    return ElasticNet(**params)


def _bayesian_ridge(params: Dict[str, Any]) -> Any:
    from sklearn.linear_model import BayesianRidge

    return BayesianRidge(**params)


def _huber(params: Dict[str, Any]) -> Any:
    from sklearn.linear_model import HuberRegressor

    return HuberRegressor(**params)


def _svr(params: Dict[str, Any]) -> Any:
    from sklearn.svm import SVR

    return SVR(**params)


def _knn(params: Dict[str, Any]) -> Any:
    from sklearn.neighbors import KNeighborsRegressor

    return KNeighborsRegressor(**params)


def _random_forest(params: Dict[str, Any]) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**params)


def _extra_trees(params: Dict[str, Any]) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(**params)


def _gbrt(params: Dict[str, Any]) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(**params)


def _hist_gbm(params: Dict[str, Any]) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(**params)


def _ada_boost(params: Dict[str, Any]) -> Any:
    from sklearn.ensemble import AdaBoostRegressor

    return AdaBoostRegressor(**params)


def _mlp(params: Dict[str, Any]) -> Any:
    from sklearn.neural_network import MLPRegressor

    return MLPRegressor(**params)


def _xgboost(params: Dict[str, Any]) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(**params)


def _lightgbm(params: Dict[str, Any]) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**params)


def _catboost(params: Dict[str, Any]) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**params)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


MODEL_SPECS: Dict[str, ModelSpec] = {
    "linear_regression": ModelSpec(
        id="linear_regression",
        label="Linear Regression",
        description="OLS baseline without regularization.",
        factory=_linear_regression,
        presets=[
            ModelPreset(
                name="default",
                params={"fit_intercept": True},
                description="Plain least squares.",
            )
        ],
    ),
    "ridge": ModelSpec(
        id="ridge",
        label="Ridge Regression",
        description="L2-regularized linear model.",
        factory=_ridge,
        presets=[
            ModelPreset(
                name="default",
                params={"alpha": 1.0, "fit_intercept": True},
                description="Light regularization.",
            ),
            ModelPreset(
                name="strong",
                params={"alpha": 10.0, "fit_intercept": True},
                description="Heavier regularization.",
            ),
        ],
    ),
    "lasso": ModelSpec(
        id="lasso",
        label="Lasso Regression",
        description="L1-regularized linear model.",
        factory=_lasso,
        presets=[
            ModelPreset(
                name="default",
                params={"alpha": 0.0005, "max_iter": 2000},
                description="Sparse weights with mild shrinkage.",
            ),
            ModelPreset(
                name="sparse",
                params={"alpha": 0.001, "max_iter": 3000},
                description="Stronger sparsity.",
            ),
        ],
    ),
    "elastic_net": ModelSpec(
        id="elastic_net",
        label="Elastic Net",
        description="Combined L1/L2 regularization.",
        factory=_elastic_net,
        presets=[
            ModelPreset(
                name="default",
                params={"alpha": 0.001, "l1_ratio": 0.2, "max_iter": 2000},
                description="Mild elastic net mix.",
            ),
            ModelPreset(
                name="sparse",
                params={"alpha": 0.002, "l1_ratio": 0.5, "max_iter": 3000},
                description="More L1 emphasis.",
            ),
        ],
    ),
    "bayesian_ridge": ModelSpec(
        id="bayesian_ridge",
        label="Bayesian Ridge",
        description="Bayesian linear regression with automatic regularization.",
        factory=_bayesian_ridge,
        presets=[
            ModelPreset(
                name="default",
                params={"fit_intercept": True, "tol": 1e-3},
                description="Stable Bayesian linear baseline.",
            ),
            ModelPreset(
                name="robust",
                params={"fit_intercept": True, "tol": 5e-4, "alpha_1": 1e-6, "lambda_1": 1e-6},
                description="Stronger priors with tighter tolerance.",
            ),
        ],
    ),
    "huber": ModelSpec(
        id="huber",
        label="Huber Regressor",
        description="Robust linear regression with outlier resistance.",
        factory=_huber,
        presets=[
            ModelPreset(
                name="default",
                params={"epsilon": 1.35, "alpha": 0.0001, "max_iter": 200},
                description="Balanced robustness.",
            ),
            ModelPreset(
                name="robust",
                params={"epsilon": 1.8, "alpha": 0.0001, "max_iter": 200},
                description="More outlier tolerance.",
            ),
        ],
    ),
    "svr": ModelSpec(
        id="svr",
        label="Support Vector Regression",
        description="Kernel-based regression for nonlinear patterns.",
        factory=_svr,
        presets=[
            ModelPreset(
                name="rbf",
                params={"C": 3.0, "epsilon": 0.1, "kernel": "rbf", "gamma": "scale"},
                description="RBF kernel with moderate regularization.",
            ),
            ModelPreset(
                name="linear",
                params={"C": 1.0, "epsilon": 0.05, "kernel": "linear"},
                description="Fast linear SVR baseline.",
            ),
        ],
    ),
    "knn": ModelSpec(
        id="knn",
        label="KNN Regressor",
        description="Nearest-neighbor regression for local structure.",
        factory=_knn,
        presets=[
            ModelPreset(
                name="default",
                params={"n_neighbors": 30, "weights": "distance", "p": 2},
                description="Distance-weighted neighbors.",
            ),
            ModelPreset(
                name="fast",
                params={"n_neighbors": 15, "weights": "distance", "p": 1},
                description="Smaller neighborhood with L1 distance.",
            ),
        ],
    ),
    "random_forest": ModelSpec(
        id="random_forest",
        label="Random Forest",
        description="Bagged decision trees for nonlinear signals.",
        factory=_random_forest,
        supports_random_state=True,
        presets=[
            ModelPreset(
                name="default",
                params={
                    "n_estimators": 400,
                    "max_depth": 8,
                    "min_samples_leaf": 20,
                    "n_jobs": -1,
                },
                description="Balanced depth and bagging.",
            ),
            ModelPreset(
                name="deep",
                params={
                    "n_estimators": 600,
                    "max_depth": 12,
                    "min_samples_leaf": 10,
                    "n_jobs": -1,
                },
                description="Deeper trees with more estimators.",
            ),
        ],
    ),
    "extra_trees": ModelSpec(
        id="extra_trees",
        label="Extra Trees",
        description="Randomized trees with stronger variance reduction.",
        factory=_extra_trees,
        supports_random_state=True,
        presets=[
            ModelPreset(
                name="default",
                params={
                    "n_estimators": 500,
                    "max_depth": 10,
                    "min_samples_leaf": 15,
                    "n_jobs": -1,
                },
                description="Fast, diversified ensembles.",
            ),
            ModelPreset(
                name="deep",
                params={
                    "n_estimators": 700,
                    "max_depth": 14,
                    "min_samples_leaf": 10,
                    "n_jobs": -1,
                },
                description="Higher capacity trees.",
            ),
        ],
    ),
    "gbrt": ModelSpec(
        id="gbrt",
        label="Gradient Boosting",
        description="Boosted trees with shrinkage.",
        factory=_gbrt,
        supports_random_state=True,
        presets=[
            ModelPreset(
                name="default",
                params={
                    "n_estimators": 400,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "subsample": 0.7,
                },
                description="Conservative boosting.",
            ),
            ModelPreset(
                name="deep",
                params={
                    "n_estimators": 600,
                    "learning_rate": 0.04,
                    "max_depth": 4,
                    "subsample": 0.8,
                },
                description="Deeper boosted trees.",
            ),
        ],
    ),
    "hist_gbm": ModelSpec(
        id="hist_gbm",
        label="Histogram GBM",
        description="Faster gradient boosting with binning.",
        factory=_hist_gbm,
        supports_random_state=True,
        presets=[
            ModelPreset(
                name="default",
                params={
                    "max_depth": 6,
                    "max_iter": 300,
                    "learning_rate": 0.05,
                    "l2_regularization": 1.0,
                    "max_bins": 255,
                    "early_stopping": True,
                },
                description="Balanced HistGBM config.",
            ),
            ModelPreset(
                name="deep",
                params={
                    "max_depth": 8,
                    "max_iter": 400,
                    "learning_rate": 0.03,
                    "l2_regularization": 0.8,
                    "max_bins": 255,
                    "early_stopping": True,
                },
                description="Higher capacity HistGBM.",
            ),
        ],
    ),
    "ada_boost": ModelSpec(
        id="ada_boost",
        label="AdaBoost",
        description="Adaptive boosting for noisy targets.",
        factory=_ada_boost,
        supports_random_state=True,
        presets=[
            ModelPreset(
                name="default",
                params={"n_estimators": 300, "learning_rate": 0.05, "loss": "square"},
                description="Stable boosting.",
            ),
            ModelPreset(
                name="fast",
                params={"n_estimators": 200, "learning_rate": 0.1, "loss": "linear"},
                description="Faster, less conservative.",
            ),
        ],
    ),
    "mlp": ModelSpec(
        id="mlp",
        label="MLP Regressor",
        description="Neural network baseline with dense layers.",
        factory=_mlp,
        supports_random_state=True,
        presets=[
            ModelPreset(
                name="default",
                params={
                    "hidden_layer_sizes": (256, 128),
                    "activation": "relu",
                    "alpha": 1e-4,
                    "learning_rate_init": 0.001,
                    "max_iter": 200,
                    "early_stopping": True,
                },
                description="Moderate MLP capacity.",
            ),
            ModelPreset(
                name="deep",
                params={
                    "hidden_layer_sizes": (512, 256, 128),
                    "activation": "relu",
                    "alpha": 5e-5,
                    "learning_rate_init": 0.0007,
                    "max_iter": 260,
                    "early_stopping": True,
                },
                description="Deeper MLP with more layers.",
            ),
        ],
    ),
}


if _has_module("xgboost"):
    MODEL_SPECS["xgboost"] = ModelSpec(
        id="xgboost",
        label="XGBoost",
        description="Gradient boosting with optimized tree splits.",
        factory=_xgboost,
        supports_random_state=True,
        presets=[
            ModelPreset(
                name="default",
                params={
                    "n_estimators": 800,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                    "n_jobs": -1,
                    "verbosity": 0,
                },
                description="Balanced XGBoost preset.",
            ),
            ModelPreset(
                name="deep",
                params={
                    "n_estimators": 1200,
                    "max_depth": 8,
                    "learning_rate": 0.03,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                    "n_jobs": -1,
                    "verbosity": 0,
                },
                description="Higher capacity XGBoost.",
            ),
        ],
    )


if _has_module("lightgbm"):
    MODEL_SPECS["lightgbm"] = ModelSpec(
        id="lightgbm",
        label="LightGBM",
        description="Leaf-wise gradient boosting with fast histogram bins.",
        factory=_lightgbm,
        supports_random_state=True,
        presets=[
            ModelPreset(
                name="default",
                params={
                    "n_estimators": 1000,
                    "learning_rate": 0.03,
                    "num_leaves": 64,
                    "max_depth": -1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "min_child_samples": 50,
                    "n_jobs": -1,
                    "verbosity": -1,
                },
                description="Balanced LightGBM preset.",
            ),
            ModelPreset(
                name="deep",
                params={
                    "n_estimators": 1400,
                    "learning_rate": 0.02,
                    "num_leaves": 128,
                    "max_depth": -1,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 1.0,
                    "min_child_samples": 30,
                    "n_jobs": -1,
                    "verbosity": -1,
                },
                description="Higher capacity LightGBM.",
            ),
        ],
    )


if _has_module("catboost"):
    MODEL_SPECS["catboost"] = ModelSpec(
        id="catboost",
        label="CatBoost",
        description="Gradient boosting with strong regularization.",
        factory=_catboost,
        seed_param="random_seed",
        presets=[
            ModelPreset(
                name="default",
                params={
                    "iterations": 900,
                    "learning_rate": 0.05,
                    "depth": 8,
                    "loss_function": "RMSE",
                    "verbose": False,
                },
                description="Balanced CatBoost preset.",
            ),
            ModelPreset(
                name="deep",
                params={
                    "iterations": 1200,
                    "learning_rate": 0.03,
                    "depth": 10,
                    "loss_function": "RMSE",
                    "verbose": False,
                },
                description="Higher capacity CatBoost.",
            ),
        ],
    )


def get_model_spec(model_id: str) -> ModelSpec:
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Unknown model id: {model_id}")
    return MODEL_SPECS[model_id]


def list_model_specs() -> List[ModelSpec]:
    return list(MODEL_SPECS.values())


def build_model(model_id: str, params: Dict[str, Any], seed: int | None) -> Any:
    spec = get_model_spec(model_id)
    merged = dict(params)
    if seed is not None:
        if spec.seed_param and spec.seed_param not in merged:
            merged[spec.seed_param] = seed
        elif spec.supports_random_state and "random_state" not in merged:
            merged["random_state"] = seed
    return spec.factory(merged)


def available_models_payload() -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for spec in list_model_specs():
        presets = [
            {
                "name": preset.name,
                "description": preset.description,
                "params": preset.params,
            }
            for preset in spec.presets
        ]
        payload.append(
            {
                "id": spec.id,
                "label": spec.label,
                "description": spec.description,
                "presets": presets,
                "default_preset": spec.presets[0].name if spec.presets else None,
            }
        )
    return payload
