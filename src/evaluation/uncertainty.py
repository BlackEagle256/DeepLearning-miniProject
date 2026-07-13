"""Uncertainty quantification (Phase 7).

Two kinds of prediction intervals (PI), as required:
  * GPR / Kriging  -> analytical PI from the posterior predictive std.
  * All other models -> Bootstrap PI (resample the training set, refit,
    collect the prediction distribution).

Reported quantities per model: PI width (mean) and empirical coverage,
which feed both the Uncertainty comparison and the multi-criteria Top-3
selection ("which models provide more trustworthy uncertainty?").
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import clone

from src.config import load_config


def gpr_prediction_interval(
    fitted_pipeline,
    X: np.ndarray,
    confidence: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical PI for a fitted GPR pipeline (scaler + GaussianProcessRegressor).

    Returns (y_pred, lower, upper).
    """
    confidence = confidence or load_config()["uncertainty"]["confidence_level"]
    z = stats.norm.ppf(0.5 + confidence / 2.0)

    scaler = fitted_pipeline.named_steps["scaler"]
    gpr = fitted_pipeline.named_steps["model"]
    X_scaled = scaler.transform(np.asarray(X))
    y_pred, y_std = gpr.predict(X_scaled, return_std=True)
    return y_pred, y_pred - z * y_std, y_pred + z * y_std


def bootstrap_prediction_interval(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    n_bootstrap: int | None = None,
    confidence: float | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile bootstrap PI for any estimator/pipeline.

    Returns (y_pred_mean, lower, upper) computed from ``n_bootstrap``
    refits on resampled training sets.
    """
    ucfg = load_config()["uncertainty"]
    n_bootstrap = n_bootstrap or ucfg["bootstrap_iterations"]
    confidence = confidence or ucfg["confidence_level"]
    alpha = (1.0 - confidence) / 2.0

    rng = np.random.default_rng(seed)
    X_train, y_train, X_eval = map(np.asarray, (X_train, y_train, X_eval))
    n = len(X_train)

    preds = np.empty((n_bootstrap, len(X_eval)))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        model = clone(estimator)
        model.fit(X_train[idx], y_train[idx])
        preds[b] = np.ravel(model.predict(X_eval))

    lower = np.quantile(preds, alpha, axis=0)
    upper = np.quantile(preds, 1.0 - alpha, axis=0)
    return preds.mean(axis=0), lower, upper


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    """Mean PI width and empirical coverage (fraction of truths inside the PI)."""
    y_true = np.ravel(np.asarray(y_true))
    width = float(np.mean(upper - lower))
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    return {"pi_mean_width": width, "pi_coverage": coverage}
