"""Evaluation metrics required by the assignment.

Per-fold metrics : R2, RMSE, NRMSE, MAE (train and test).
Aggregates       : mean/std over folds and seeds,
                   generalization gap = |train - test| distance,
                   fold stability    = std of test metric across folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE normalized by the range of the true values (range-NRMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    rng = float(np.max(y_true) - np.min(y_true))
    if rng == 0.0:
        return float("nan")
    return rmse(y_true, y_pred) / rng


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute the four core regression metrics with an optional key prefix."""
    return {
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}nrmse": nrmse(y_true, y_pred),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
    }


def generalization_gap(train_metric: float, test_metric: float) -> float:
    """Train/Test distance - the primary overfitting indicator of the project."""
    return float(abs(train_metric - test_metric))


def aggregate_fold_results(fold_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate per-fold results.

    ``fold_df`` must contain columns like ``train_r2 / test_r2 / ...`` plus
    the grouping columns (e.g. model, target, seed). Returns mean and std of
    every metric plus the R2 and RMSE generalization gaps and fold-stability
    (std of test metrics), which feed the multi-criteria Top-3 selection.
    """
    metric_cols = [c for c in fold_df.columns if c.startswith(("train_", "test_"))]
    agg = fold_df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()

    agg["gap_r2"] = (agg["train_r2_mean"] - agg["test_r2_mean"]).abs()
    agg["gap_rmse"] = (agg["train_rmse_mean"] - agg["test_rmse_mean"]).abs()
    # Fold stability: lower std across folds/seeds = more stable model.
    agg["stability_test_r2_std"] = agg["test_r2_std"]
    agg["stability_test_rmse_std"] = agg["test_rmse_std"]
    return agg
