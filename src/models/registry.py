"""Model registry - single source of truth for all project models.

Models (Section 2.1 / Phase 3 of the assignment):
    1. Linear Regression (BASELINE)   2. Ridge          3. ElasticNet
    4. SVR-RBF                        5. GPR (Kriging)  6. Random Forest
    7. Extra Trees                    8. Gradient Boosting
    9. XGBoost                       10. shallow ANN (MLPRegressor)
   11. LightGBM (extra, listed in the Phase-3 pipeline model list)

Design rules enforced here:
  * Every model is wrapped in a sklearn ``Pipeline`` whose FIRST step is a
    ``StandardScaler``. Because scaling lives INSIDE the pipeline, it is
    fitted only on the training part of each CV fold -> no data leakage.
  * The ANN is intentionally shallow: 1 hidden layer, 8-32 neurons, ReLU,
    Adam, with early stopping. Deep ANNs are forbidden by the assignment.
  * Overfitting management (Table 2) is baked into the default
    hyperparameters: L1/L2 regularization, early stopping, shallow trees /
    min_samples_leaf, kernel length-scale bounds, etc.
  * Random-Search spaces for every model live in ``SEARCH_SPACES``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

BASELINE_MODEL = "linear_regression"

# Models whose sklearn estimator natively supports multi-output regression.
NATIVE_MULTIOUTPUT = {"linear_regression", "ridge", "random_forest", "extra_trees", "gpr", "ann"}


def _gpr_kernel() -> object:
    """Kriging kernel: constant * RBF + white noise.

    Length-scale bounds act as kernel regularization (Table 2), and the
    WhiteKernel absorbs experimental noise, both fighting overfitting.
    """
    return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))


def _estimator_factories(seed: int) -> dict[str, Callable[[], object]]:
    """Return name -> factory for the bare estimators (before scaling)."""
    factories: dict[str, Callable[[], object]] = {
        "linear_regression": lambda: LinearRegression(),
        "ridge": lambda: Ridge(alpha=1.0, random_state=seed),
        "elasticnet": lambda: ElasticNet(
            alpha=0.1, l1_ratio=0.5, max_iter=20_000, random_state=seed
        ),
        "svr_rbf": lambda: SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"),
        "gpr": lambda: GaussianProcessRegressor(
            kernel=_gpr_kernel(),
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=seed,
        ),
        # Shallow trees + min_samples_leaf = anti-overfitting (Table 2).
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=2,
            subsample=0.8,
            # Early stopping via internal validation fraction (Table 2).
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=seed,
        ),
        # Shallow ANN as required: 1 hidden layer, 8-32 neurons, ReLU, Adam.
        "ann": lambda: MLPRegressor(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            alpha=1e-2,               # L2 regularization
            max_iter=5_000,
            early_stopping=True,      # Table 2: early stopping for ANN
            validation_fraction=0.2,
            n_iter_no_change=30,
            random_state=seed,
        ),
    }

    # Optional dependencies - keep the repo importable without them.
    try:
        from xgboost import XGBRegressor

        factories["xgboost"] = lambda: XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=2,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,           # L1
            reg_lambda=1.0,          # L2
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:  # pragma: no cover
        pass

    try:
        from lightgbm import LGBMRegressor

        factories["lightgbm"] = lambda: LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=3,     # tiny datasets need tiny leaves
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:  # pragma: no cover
        pass

    return factories


def available_models(seed: int = 42) -> list[str]:
    """Names of all models available in the current environment."""
    return list(_estimator_factories(seed).keys())


def build_model(name: str, seed: int = 42) -> Pipeline:
    """Build the full pipeline ``StandardScaler -> estimator`` for one model.

    Scaling inside the pipeline guarantees it is (re)fitted on the training
    fold only during cross-validation -> no data leakage.
    """
    factories = _estimator_factories(seed)
    if name not in factories:
        raise KeyError(f"Unknown or unavailable model '{name}'. Available: {list(factories)}")
    return Pipeline([("scaler", StandardScaler()), ("model", factories[name]())])


# ---------------------------------------------------------------------------
# Random Search spaces (Phase 4). Keys are pipeline-parameter names.
# Grid Search is deliberately NOT provided (forbidden for small data).
# ---------------------------------------------------------------------------
SEARCH_SPACES: dict[str, dict] = {
    "linear_regression": {},  # baseline: no tuning
    "ridge": {"model__alpha": loguniform(1e-3, 1e2)},
    "elasticnet": {
        "model__alpha": loguniform(1e-3, 1e1),
        "model__l1_ratio": uniform(0.05, 0.9),
    },
    "svr_rbf": {
        "model__C": loguniform(1e-1, 1e2),
        "model__epsilon": loguniform(1e-3, 1e0),
        "model__gamma": loguniform(1e-3, 1e1),
    },
    "gpr": {
        "model__alpha": loguniform(1e-10, 1e-1),
        "model__n_restarts_optimizer": randint(2, 10),
    },
    "random_forest": {
        "model__n_estimators": randint(100, 600),
        "model__max_depth": randint(2, 7),
        "model__min_samples_leaf": randint(1, 5),
        "model__max_features": uniform(0.4, 0.6),
    },
    "extra_trees": {
        "model__n_estimators": randint(100, 600),
        "model__max_depth": randint(2, 7),
        "model__min_samples_leaf": randint(1, 5),
        "model__max_features": uniform(0.4, 0.6),
    },
    "gradient_boosting": {
        "model__n_estimators": randint(100, 800),
        "model__learning_rate": loguniform(1e-3, 3e-1),
        "model__max_depth": randint(1, 4),
        "model__min_samples_leaf": randint(1, 5),
        "model__subsample": uniform(0.6, 0.4),
    },
    "xgboost": {
        "model__n_estimators": randint(100, 800),
        "model__learning_rate": loguniform(1e-3, 3e-1),
        "model__max_depth": randint(1, 4),
        "model__min_child_weight": randint(1, 6),
        "model__subsample": uniform(0.6, 0.4),
        "model__reg_alpha": loguniform(1e-3, 1e1),
        "model__reg_lambda": loguniform(1e-2, 1e1),
    },
    "lightgbm": {
        "model__n_estimators": randint(100, 800),
        "model__learning_rate": loguniform(1e-3, 3e-1),
        "model__max_depth": randint(2, 5),
        "model__num_leaves": randint(3, 15),
        "model__min_child_samples": randint(2, 6),
        "model__reg_alpha": loguniform(1e-3, 1e1),
        "model__reg_lambda": loguniform(1e-2, 1e1),
    },
    "ann": {
        "model__hidden_layer_sizes": [(h,) for h in (8, 12, 16, 24, 32)],
        "model__alpha": loguniform(1e-4, 1e0),
        "model__learning_rate_init": loguniform(1e-4, 1e-2),
    },
}
