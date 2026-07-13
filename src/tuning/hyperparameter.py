"""Phase 4 - Hyperparameter optimization.

Policy (from the assignment):
  * Random Search for ALL models.
  * Bayesian Optimization (Optuna) ONLY for the Top-3 models, after the
    first stage.
  * Grid Search is FORBIDDEN (dataset is too small) - deliberately not
    implemented here.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV

from src.config import load_config
from src.models.registry import SEARCH_SPACES, build_model


def random_search(model_name: str, X, y, seed: int = 42) -> RandomizedSearchCV:
    """Randomized search over the registry's space for one model."""
    cfg = load_config()
    space = SEARCH_SPACES.get(model_name, {})
    cv = KFold(n_splits=cfg["cross_validation"]["n_splits"], shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        build_model(model_name, seed=seed),
        param_distributions=space,
        n_iter=cfg["tuning"]["random_search_iterations"] if space else 1,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(np.asarray(X), np.asarray(y))
    return search


def optuna_tune(model_name: str, X, y, seed: int = 42, n_trials: int | None = None):
    """Bayesian optimization with Optuna - Top-3 models only.

    Returns the Optuna study; ``study.best_params`` holds the winning
    pipeline parameters (same naming as the Random-Search space).
    """
    import optuna
    from sklearn.model_selection import cross_val_score

    cfg = load_config()
    n_trials = n_trials or cfg["tuning"]["optuna_trials"]
    cv = KFold(n_splits=cfg["cross_validation"]["n_splits"], shuffle=True, random_state=seed)
    X_np, y_np = np.asarray(X), np.asarray(y)

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_params(trial, model_name)
        pipe = clone(build_model(model_name, seed=seed)).set_params(**params)
        scores = cross_val_score(
            pipe, X_np, y_np, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def _suggest_params(trial, model_name: str) -> dict:
    """Optuna search spaces mirroring the Random-Search ranges."""
    if model_name == "ridge":
        return {"model__alpha": trial.suggest_float("model__alpha", 1e-3, 1e2, log=True)}
    if model_name == "elasticnet":
        return {
            "model__alpha": trial.suggest_float("model__alpha", 1e-3, 1e1, log=True),
            "model__l1_ratio": trial.suggest_float("model__l1_ratio", 0.05, 0.95),
        }
    if model_name == "svr_rbf":
        return {
            "model__C": trial.suggest_float("model__C", 1e-1, 1e2, log=True),
            "model__epsilon": trial.suggest_float("model__epsilon", 1e-3, 1.0, log=True),
            "model__gamma": trial.suggest_float("model__gamma", 1e-3, 1e1, log=True),
        }
    if model_name == "gpr":
        return {"model__alpha": trial.suggest_float("model__alpha", 1e-10, 1e-1, log=True)}
    if model_name in ("random_forest", "extra_trees"):
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 600),
            "model__max_depth": trial.suggest_int("model__max_depth", 2, 6),
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 4),
        }
    if model_name == "gradient_boosting":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 800),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 1e-3, 0.3, log=True),
            "model__max_depth": trial.suggest_int("model__max_depth", 1, 3),
            "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0),
        }
    if model_name == "xgboost":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 800),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 1e-3, 0.3, log=True),
            "model__max_depth": trial.suggest_int("model__max_depth", 1, 3),
            "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-3, 1e1, log=True),
            "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-2, 1e1, log=True),
            "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0),
        }
    if model_name == "lightgbm":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 800),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 1e-3, 0.3, log=True),
            "model__num_leaves": trial.suggest_int("model__num_leaves", 3, 15),
            "model__min_child_samples": trial.suggest_int("model__min_child_samples", 2, 6),
        }
    if model_name == "ann":
        return {
            "model__hidden_layer_sizes": (trial.suggest_int("hidden_units", 8, 32),),
            "model__alpha": trial.suggest_float("model__alpha", 1e-4, 1.0, log=True),
            "model__learning_rate_init": trial.suggest_float(
                "model__learning_rate_init", 1e-4, 1e-2, log=True
            ),
        }
    return {}  # linear_regression baseline: nothing to tune
