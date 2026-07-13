"""Phase 2 - outlier DETECTION (never removal).

Methods (all three required): IQR rule, Local Outlier Factor,
Isolation Forest. Outliers are ONLY flagged and reported - in mechanics
they may reflect true physical behaviour, so no row is ever dropped
without a mechanical justification (and none is dropped in this project).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def iqr_outliers(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Per-column IQR flags; a row is flagged if ANY column is outside fences."""
    num = df.select_dtypes("number")
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    mask = (num < (q1 - k * iqr)) | (num > (q3 + k * iqr))
    out = mask.copy()
    out["iqr_outlier_any"] = mask.any(axis=1)
    return out


def lof_outliers(df: pd.DataFrame, n_neighbors: int = 10, seed: int = 42) -> pd.Series:
    """Local Outlier Factor flags (True = outlier). Data is standardized first."""
    _ = seed  # LOF is deterministic; kept for a uniform signature
    X = StandardScaler().fit_transform(df.select_dtypes("number"))
    n_neighbors = min(n_neighbors, len(df) - 1)
    labels = LocalOutlierFactor(n_neighbors=n_neighbors).fit_predict(X)
    return pd.Series(labels == -1, index=df.index, name="lof_outlier")


def isolation_forest_outliers(
    df: pd.DataFrame, contamination: float = 0.1, seed: int = 42
) -> pd.Series:
    """Isolation Forest flags (True = outlier)."""
    X = StandardScaler().fit_transform(df.select_dtypes("number"))
    labels = IsolationForest(contamination=contamination, random_state=seed).fit_predict(X)
    return pd.Series(labels == -1, index=df.index, name="iforest_outlier")


def outlier_report(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Combined report: IQR any-flag, LOF flag, IForest flag, consensus count."""
    report = pd.DataFrame(index=df.index)
    report["iqr_outlier"] = iqr_outliers(df)["iqr_outlier_any"]
    report["lof_outlier"] = lof_outliers(df, seed=seed)
    report["iforest_outlier"] = isolation_forest_outliers(df, seed=seed)
    report["n_methods_flagging"] = report[
        ["iqr_outlier", "lof_outlier", "iforest_outlier"]
    ].sum(axis=1)
    return report
