"""Overfitting detection and monitoring (the core theme of the project).

Tools:
  * learning curves      -> does more data close the train/test gap?
  * validation curves    -> complexity sweep for a chosen hyperparameter
  * gap report           -> tabulated Train/Test distance per model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, learning_curve, validation_curve

from src.config import load_config


def compute_learning_curve(estimator, X, y, seed: int = 42, n_points: int = 6) -> pd.DataFrame:
    """Learning curve with negative-RMSE scoring, CV per the global config."""
    cv = KFold(
        n_splits=load_config()["cross_validation"]["n_splits"],
        shuffle=True,
        random_state=seed,
    )
    sizes, train_scores, test_scores = learning_curve(
        estimator,
        np.asarray(X),
        np.asarray(y),
        train_sizes=np.linspace(0.3, 1.0, n_points),
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        shuffle=True,
        random_state=seed,
    )
    return pd.DataFrame(
        {
            "train_size": sizes,
            "train_rmse_mean": -train_scores.mean(axis=1),
            "train_rmse_std": train_scores.std(axis=1),
            "test_rmse_mean": -test_scores.mean(axis=1),
            "test_rmse_std": test_scores.std(axis=1),
        }
    )


def compute_validation_curve(
    estimator, X, y, param_name: str, param_range, seed: int = 42
) -> pd.DataFrame:
    """Validation curve over one complexity hyperparameter (e.g. max_depth)."""
    cv = KFold(
        n_splits=load_config()["cross_validation"]["n_splits"],
        shuffle=True,
        random_state=seed,
    )
    train_scores, test_scores = validation_curve(
        estimator,
        np.asarray(X),
        np.asarray(y),
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "param_value": list(param_range),
            "train_rmse_mean": -train_scores.mean(axis=1),
            "test_rmse_mean": -test_scores.mean(axis=1),
        }
    )


def plot_learning_curve(curve_df: pd.DataFrame, title: str):
    """Standard learning-curve figure (returned, not shown)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve_df["train_size"], curve_df["train_rmse_mean"], "o-", label="Train RMSE")
    ax.plot(curve_df["train_size"], curve_df["test_rmse_mean"], "s-", label="CV RMSE")
    ax.fill_between(
        curve_df["train_size"],
        curve_df["test_rmse_mean"] - curve_df["test_rmse_std"],
        curve_df["test_rmse_mean"] + curve_df["test_rmse_std"],
        alpha=0.2,
    )
    ax.set_xlabel("Training set size")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
