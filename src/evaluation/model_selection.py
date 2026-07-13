"""Multi-criteria Top-3 model selection (Phase 5).

Accuracy alone must NOT decide the best models. The composite score
combines (with config weights):
  * accuracy            : mean test R2 (higher better)
  * generalization gap  : |train R2 - test R2| (lower better)
  * fold stability      : std of test R2 across folds/seeds (lower better)
  * uncertainty         : mean PI width (lower better; optional column)
  * interpretability    : ordinal prior from the config (lower rank better)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config


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
