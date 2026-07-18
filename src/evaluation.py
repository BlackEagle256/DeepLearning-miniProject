"""Unified evaluation module (merged from the former ``src/evaluation`` package).

Contents:
  1. Metrics             : R2, RMSE, NRMSE, MAE per fold + aggregation,
                           generalization gap, fold stability.
  2. Model selection     : multi-criteria Top-3 composite score (Phase 5).
  3. Overfitting         : learning curves, validation curves, gap monitoring.
  4. Statistical tests   : Friedman + Nemenyi post-hoc, pairwise Wilcoxon (Phase 7).
  5. Uncertainty         : analytical GPR prediction intervals and bootstrap
                           prediction intervals, PI width / coverage (Phase 7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, learning_curve, validation_curve

from src.config import load_config

# ---------------------------------------------------------------------------
# 1. Metrics
#
# Per-fold metrics : R2, RMSE, NRMSE, MAE (train and test).
# Aggregates       : mean/std over folds and seeds,
#                    generalization gap = |train - test| distance,
#                    fold stability    = std of test metric across folds.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 2. Multi-criteria Top-3 model selection (Phase 5)
#
# Accuracy alone must NOT decide the best models. The composite score
# combines (with config weights):
#   * accuracy            : mean test R2 (higher better)
#   * generalization gap  : |train R2 - test R2| (lower better)
#   * fold stability      : std of test R2 across folds/seeds (lower better)
#   * uncertainty         : mean PI width (lower better; optional column)
#   * interpretability    : ordinal prior from the config (lower rank better)
# ---------------------------------------------------------------------------


def _minmax(s: pd.Series, higher_is_better: bool) -> pd.Series:
    """Normalize to [0, 1] where 1 is always 'better'."""
    rng = s.max() - s.min()
    norm = (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)
    return norm if higher_is_better else 1.0 - norm


def select_top3(agg_df: pd.DataFrame, pi_width_col: str | None = None) -> pd.DataFrame:
    """Rank models by the weighted multi-criteria score.

    ``agg_df`` must contain one row per model with columns:
      model, test_r2_mean, gap_r2, stability_test_r2_std
    and optionally a PI-width column (``pi_width_col``).
    Returns the frame sorted by composite score with a ``rank`` column;
    take ``.head(3)`` for the Top-3.
    """
    cfg = load_config()["top3_selection"]
    w = cfg["weights"]
    interp_rank = cfg["interpretability_rank"]

    df = agg_df.copy()
    scores = pd.DataFrame(index=df.index)
    scores["accuracy"] = _minmax(df["test_r2_mean"], higher_is_better=True)
    scores["generalization_gap"] = _minmax(df["gap_r2"], higher_is_better=False)
    scores["fold_stability"] = _minmax(df["stability_test_r2_std"], higher_is_better=False)

    if pi_width_col and pi_width_col in df.columns:
        scores["uncertainty"] = _minmax(df[pi_width_col], higher_is_better=False)
    else:
        scores["uncertainty"] = 0.5  # neutral when PI widths are not yet computed

    ranks = df["model"].map(lambda m: interp_rank.get(m, np.median(list(interp_rank.values()))))
    scores["interpretability"] = _minmax(pd.Series(ranks, index=df.index), higher_is_better=False)

    df["composite_score"] = sum(w[k] * scores[k] for k in w)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ---------------------------------------------------------------------------
# 3. Overfitting detection and monitoring (the core theme of the project)
#
# Tools:
#   * learning curves      -> does more data close the train/test gap?
#   * validation curves    -> complexity sweep for a chosen hyperparameter
#   * gap report           -> tabulated Train/Test distance per model
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 4. Statistical model comparison (Phase 7)
#
# Friedman test + Nemenyi post-hoc (or pairwise Wilcoxon signed-rank) over
# per-fold scores, used to decide whether differences between models - and
# between Single-output and Multi-output modes - are statistically
# significant.
# ---------------------------------------------------------------------------


def friedman_test(score_matrix: pd.DataFrame) -> dict[str, float]:
    """Friedman test.

    ``score_matrix``: rows = blocks (e.g. fold x seed), columns = models,
    values = a test metric (e.g. test RMSE).
    """
    statistic, p_value = stats.friedmanchisquare(
        *[score_matrix[c].to_numpy() for c in score_matrix.columns]
    )
    return {"statistic": float(statistic), "p_value": float(p_value)}


def nemenyi_posthoc(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """Nemenyi post-hoc pairwise p-values (requires scikit-posthocs)."""
    import scikit_posthocs as sp

    return sp.posthoc_nemenyi_friedman(score_matrix.to_numpy())


def wilcoxon_pairwise(scores_a: pd.Series, scores_b: pd.Series) -> dict[str, float]:
    """Wilcoxon signed-rank between two paired score vectors.

    Typical use: Single-output vs Multi-output scores of the same model on
    the same folds (paired design).
    """
    statistic, p_value = stats.wilcoxon(scores_a.to_numpy(), scores_b.to_numpy())
    return {"statistic": float(statistic), "p_value": float(p_value)}


# ---------------------------------------------------------------------------
# 5. Uncertainty quantification (Phase 7)
#
# Two kinds of prediction intervals (PI), as required:
#   * GPR / Kriging  -> analytical PI from the posterior predictive std.
#   * All other models -> Bootstrap PI (resample the training set, refit,
#     collect the prediction distribution).
#
# Reported quantities per model: PI width (mean) and empirical coverage,
# which feed both the Uncertainty comparison and the multi-criteria Top-3
# selection ("which models provide more trustworthy uncertainty?").
# ---------------------------------------------------------------------------


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
