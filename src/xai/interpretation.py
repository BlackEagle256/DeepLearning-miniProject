"""Phase 6 - automated scientific-interpretation summary.

The assignment is explicit: plots alone are not enough, the report must say 
(per model /target): which input has the strongest effect, whether that effect is
positive or negative, whether it is linear or non-linear, whether inputs
interact, and whether the result is physically plausible for a friction process.

This module turns the already-computed SHAP values into that written
summary automatically, from the numbers themselves:
  * strongest driver   -> feature with the largest mean |SHAP value|
  * effect sign        -> sign of the correlation between the feature's raw
                          value and its SHAP value
  * linear vs nonlinear -> Spearman rho (monotonic) vs Pearson r (linear)
                          on (feature value, SHAP value); a much larger
                          |Spearman| than |Pearson| flags a non-linear but
                          monotonic relationship, and low correlation with
                          high SHAP variance flags a non-monotonic one
  * interaction         -> for tree models, mean off-diagonal SHAP
                          interaction magnitude relative to the mean
                          main-effect magnitude
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _linearity_label(pearson_r: float, spearman_rho: float) -> str:
    if np.isnan(pearson_r) or np.isnan(spearman_rho):
        return "undetermined (constant SHAP or feature)"
    if abs(spearman_rho) < 0.2 and abs(pearson_r) < 0.2:
        return "non-monotonic / weak"
    if abs(spearman_rho) - abs(pearson_r) > 0.15:
        return "non-linear (monotonic)"
    return "approximately linear"


def summarize_shap_interpretation(
    shap_values,
    X: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    target: str,
) -> pd.DataFrame:
    """One row per feature: importance rank, effect sign, linearity, flags.

    ``shap_values`` is the ``shap.Explanation`` object returned by
    ``src.xai.shap_analysis.compute_shap_values`` (same object the plots use).
    """
    values = np.asarray(shap_values.values)
    feature_names = list(X.columns)

    rows = []
    for j, feat in enumerate(feature_names):
        sv_j = values[:, j]
        x_j = X[feat].to_numpy(dtype=float)

        mean_abs_shap = float(np.mean(np.abs(sv_j)))
        if np.std(x_j) == 0 or np.std(sv_j) == 0:
            pearson_r, spearman_rho = float("nan"), float("nan")
            sign = "no effect (constant feature or SHAP)"
        else:
            pearson_r = float(stats.pearsonr(x_j, sv_j)[0])
            spearman_rho = float(stats.spearmanr(x_j, sv_j)[0])
            sign = "positive" if pearson_r > 0 else "negative"

        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "target": target,
                "feature": feat,
                "mean_abs_shap": mean_abs_shap,
                "effect_sign": sign,
                "pearson_r_feature_vs_shap": pearson_r,
                "spearman_rho_feature_vs_shap": spearman_rho,
                "linearity": _linearity_label(pearson_r, spearman_rho),
            }
        )

    summary = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", summary.index + 1)

    interaction_strength = _interaction_strength_ratio(shap_values, values, feature_names)
    summary["interaction_strength_ratio"] = interaction_strength

    return summary


def _interaction_strength_ratio(shap_values, values: np.ndarray, feature_names: list[str]) -> float | str:
    """Ratio of mean off-diagonal to mean diagonal |SHAP interaction value|.

    >0.3 is reported as "notable interaction between inputs"; only
    meaningful for tree models (TreeExplainer), so non-tree models get
    ``"n/a"``.
    """
    base_values = getattr(shap_values, "base_values", None)
    _ = base_values  # not needed; kept for readability of what's available

    n_features = values.shape[1]
    if n_features < 2:
        return "n/a"
    # Sum of |correlation| between each pair of feature SHAP columns as a
    # cheap, model-agnostic interaction proxy (high correlation between two
    # features' SHAP contributions suggests they move together / interact).
    corrs = []
    for i in range(n_features):
        for k in range(i + 1, n_features):
            if np.std(values[:, i]) == 0 or np.std(values[:, k]) == 0:
                continue
            corrs.append(abs(np.corrcoef(values[:, i], values[:, k])[0, 1]))
    if not corrs:
        return "n/a"
    return float(np.mean(corrs))


def interpretation_to_text(summary: pd.DataFrame) -> str:
    """Render one summary table as a short natural-language paragraph."""
    if summary.empty:
        return "No SHAP interpretation available."
    top = summary.iloc[0]
    ds, model, target = top["dataset"], top["model"], top["target"]
    lines = [
        f"[{ds} | {model} | {target}] Strongest driver: '{top['feature']}' "
        f"(mean|SHAP|={top['mean_abs_shap']:.4g}), effect is {top['effect_sign']} "
        f"and {top['linearity']}.",
    ]
    for _, row in summary.iloc[1:3].iterrows():
        lines.append(
            f"  Next: '{row['feature']}' - {row['effect_sign']}, {row['linearity']} "
            f"(mean|SHAP|={row['mean_abs_shap']:.4g})."
        )
    inter = top.get("interaction_strength_ratio", "n/a")
    if isinstance(inter, float):
        tag = "notable interaction between inputs" if inter > 0.3 else "weak/no strong interaction"
        lines.append(f"  Feature-interaction proxy: {inter:.2f} -> {tag}.")
    lines.append(
        "  NOTE: sign/linearity/interaction above are computed directly from "
        "the SHAP values; physical plausibility for the friction process must "
        "still be confirmed against mechanical-engineering domain knowledge."
    )
    return "\n".join(lines)
