"""Cross-validation engine shared by all four project pipelines.

Protocol (per assignment):
  * k-Fold (k=5) as the main framework, LOOCV available as an option.
  * Every experiment repeated over 5 different random seeds (small-data
    results are seed-sensitive).
  * Scaling happens INSIDE the sklearn Pipeline -> fitted only on the
    training part of each fold (no data leakage).
Outputs per (model, target/multi, seed, fold): train and test
R2 / RMSE / NRMSE / MAE, ready for aggregation and statistical tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, LeaveOneOut

from src.config import load_config
from src.evaluation import compute_metrics
from src.utils.seeds import set_global_seed


def _make_splitter(seed: int):
    cv_cfg = load_config()["cross_validation"]
    if cv_cfg.get("use_loocv", False):
        return LeaveOneOut()
    return KFold(n_splits=cv_cfg["n_splits"], shuffle=cv_cfg["shuffle"], random_state=seed)


def cross_validate_model(
    estimator,
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Run one CV round for one (estimator, seed) pair.

    Works for both single-output (Series) and multi-output (DataFrame) ``y``.
    For LOOCV single test points, per-fold R2/NRMSE are undefined; use k-fold
    for per-fold statistics or aggregate LOOCV predictions externally.

    For multi-output ``y`` (DataFrame with >1 column), per-target metrics are
    ADDED alongside the aggregate ones (e.g. ``test_Hardness (HV)__rmse``),
    keyed as ``{train,test}_{target}__{metric}``. This is required to compare
    Multi-output against Single-output fairly: sklearn's aggregate multi-output
    RMSE is sqrt(mean of per-target MSE), which by the RMS/QM-AM inequality is
    ALWAYS >= the mean of per-target RMSEs used for Single-output - comparing
    the two aggregates directly is a scale artifact, not a real effect.
    """
    set_global_seed(seed)
    splitter = _make_splitter(seed)
    X_np, y_np = X.to_numpy(), np.asarray(y)
    target_names = list(y.columns) if isinstance(y, pd.DataFrame) and y.shape[1] > 1 else None

    rows: list[dict] = []
    for fold_idx, (tr, te) in enumerate(splitter.split(X_np)):
        model = clone(estimator)
        model.fit(X_np[tr], y_np[tr])
        y_tr_pred = model.predict(X_np[tr])
        y_te_pred = model.predict(X_np[te])

        row: dict = {"seed": seed, "fold": fold_idx}
        row.update(compute_metrics(y_np[tr], y_tr_pred, prefix="train_"))
        row.update(compute_metrics(y_np[te], y_te_pred, prefix="test_"))

        if target_names is not None:
            for j, col in enumerate(target_names):
                row.update(compute_metrics(y_np[tr][:, j], y_tr_pred[:, j], prefix=f"train_{col}__"))
                row.update(compute_metrics(y_np[te][:, j], y_te_pred[:, j], prefix=f"test_{col}__"))

        rows.append(row)

    return pd.DataFrame(rows)


def cross_validate_multi_seed(
    estimator,
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Repeat cross-validation over all configured seeds and stack results."""
    seeds = seeds if seeds is not None else load_config()["reproducibility"]["seeds"]
    return pd.concat(
        [cross_validate_model(estimator, X, y, seed=s) for s in seeds],
        ignore_index=True,
    )


def nested_cv_score(estimator, param_distributions: dict, X, y, seed: int) -> pd.DataFrame:
    """Nested CV (outer evaluation, inner Random-Search tuning).

    Used as an honest generalization estimate for tuned models (Table 2:
    Nested CV is part of the global evaluation framework).
    """
    from sklearn.model_selection import RandomizedSearchCV

    cfg = load_config()
    nested_cfg = cfg["cross_validation"]["nested"]
    n_iter = cfg["tuning"]["random_search_iterations"]

    outer = KFold(n_splits=nested_cfg["outer_splits"], shuffle=True, random_state=seed)
    X_np, y_np = np.asarray(X), np.asarray(y)

    rows = []
    for fold_idx, (tr, te) in enumerate(outer.split(X_np)):
        inner = KFold(n_splits=nested_cfg["inner_splits"], shuffle=True, random_state=seed)
        search = RandomizedSearchCV(
            clone(estimator),
            param_distributions=param_distributions,
            n_iter=n_iter if param_distributions else 1,
            cv=inner,
            scoring="neg_root_mean_squared_error",
            random_state=seed,
            n_jobs=-1,
        )
        search.fit(X_np[tr], y_np[tr])
        row = {"seed": seed, "fold": fold_idx, "best_params": str(search.best_params_)}
        row.update(compute_metrics(y_np[te], search.predict(X_np[te]), prefix="test_"))
        rows.append(row)
    return pd.DataFrame(rows)
