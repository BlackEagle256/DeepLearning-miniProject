"""Statistical model comparison (Phase 7).

Friedman test + Nemenyi post-hoc (or pairwise Wilcoxon signed-rank) over
per-fold scores, used to decide whether differences between models - and
between Single-output and Multi-output modes - are statistically
significant.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats


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
