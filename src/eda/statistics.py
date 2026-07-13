"""Phase 2 - descriptive statistics, normality and correlation tests.

Descriptive : mean, median, mode, variance, std, skewness, kurtosis
              + Kendall-tau confidence interval per variable pair.
Normality   : Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov.
Correlation : Pearson, Spearman, Kendall (coefficients + p-values).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Full descriptive-statistics table for all numeric columns."""
    num = df.select_dtypes("number")
    out = pd.DataFrame(
        {
            "mean": num.mean(),
            "median": num.median(),
            "mode": num.mode().iloc[0],
            "variance": num.var(),
            "std": num.std(),
            "skewness": num.skew(),
            "kurtosis": num.kurtosis(),
            "min": num.min(),
            "max": num.max(),
        }
    )
    return out.round(4)


def normality_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Shapiro-Wilk, Anderson-Darling and KS test per numeric column."""
    rows = []
    for col in df.select_dtypes("number").columns:
        x = df[col].dropna().to_numpy()
        sw_stat, sw_p = stats.shapiro(x)
        ad = stats.anderson(x, dist="norm")
        # KS against a normal with the sample's own mean/std
        ks_stat, ks_p = stats.kstest(x, "norm", args=(x.mean(), x.std(ddof=1)))
        rows.append(
            {
                "variable": col,
                "shapiro_stat": sw_stat,
                "shapiro_p": sw_p,
                "anderson_stat": ad.statistic,
                "anderson_crit_5pct": ad.critical_values[2],
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "normal_at_5pct(shapiro)": sw_p > 0.05,
            }
        )
    return pd.DataFrame(rows).round(4)


def correlation_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson/Spearman/Kendall coefficients with p-values."""
    cols = list(df.select_dtypes("number").columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            x, y = df[a].to_numpy(), df[b].to_numpy()
            pr, pr_p = stats.pearsonr(x, y)
            sr, sr_p = stats.spearmanr(x, y)
            kt, kt_p = stats.kendalltau(x, y)
            rows.append(
                {
                    "var_a": a, "var_b": b,
                    "pearson_r": pr, "pearson_p": pr_p,
                    "spearman_rho": sr, "spearman_p": sr_p,
                    "kendall_tau": kt, "kendall_p": kt_p,
                }
            )
    return pd.DataFrame(rows).round(4)


def kendall_tau_confidence_interval(
    x: np.ndarray, y: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 2000, seed: int = 42
) -> dict[str, float]:
    """Bootstrap confidence interval for Kendall's tau of one pair."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    taus = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        taus[b], _ = stats.kendalltau(x[idx], y[idx])
    alpha = (1.0 - confidence) / 2.0
    tau, p = stats.kendalltau(x, y)
    return {
        "tau": float(tau),
        "p_value": float(p),
        "ci_lower": float(np.nanquantile(taus, alpha)),
        "ci_upper": float(np.nanquantile(taus, 1 - alpha)),
    }


def kendall_ci_table(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Kendall tau + bootstrap CI for every numeric variable pair."""
    cols = list(df.select_dtypes("number").columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            res = kendall_tau_confidence_interval(df[a], df[b], confidence=confidence)
            rows.append({"var_a": a, "var_b": b, **res})
    return pd.DataFrame(rows).round(4)
