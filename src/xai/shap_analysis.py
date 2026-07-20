"""Phase 6 - SHAP analysis (Global + Local).

Required plots: Summary, Beeswarm, Waterfall, Dependence, Feature
Interaction, Feature Importance. All functions save figures under
``results/xai/<dataset>/<model>/`` and return the SHAP values so that the
scientific interpretation (effect sign, linearity, interactions, physical
consistency with the friction process) can be written on top of them.

NOTE: never report constant/dropped features as important - the loader
already removes them before models see the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.registry import unwrap_pipeline
from src.utils.io import ensure_dir


def _explainer_for(fitted_pipeline, X_background: pd.DataFrame):
    """Choose the right SHAP explainer for the fitted pipeline.

    Tree models -> TreeExplainer (exact, fast); everything else ->
    model-agnostic Explainer on the pipeline's predict function.

    Some XGBoost/SHAP version combinations fail inside TreeExplainer with
    a ``ValueError`` while parsing the model's ``base_score`` (stored in
    scientific notation, e.g. ``'[1.9E0]'``, by newer XGBoost releases).
    This is a known cross-version incompatibility, not a data or pipeline
    problem, so on failure we transparently fall back to the same
    model-agnostic explainer used for non-tree models instead of crashing
    the whole XAI run.
    """
    import shap

    inner = unwrap_pipeline(fitted_pipeline)
    model = inner.named_steps["model"]
    scaler = inner.named_steps["scaler"]
    tree_types = (
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "GradientBoostingRegressor",
        "XGBRegressor",
        "LGBMRegressor",
    )
    if type(model).__name__ in tree_types:
        X_bg = pd.DataFrame(
            scaler.transform(X_background), columns=X_background.columns
        )
        try:
            return shap.TreeExplainer(model), X_bg
        except Exception as exc:  # pragma: no cover - version-dependent bug
            print(
                f"[shap] TreeExplainer failed for {type(model).__name__} "
                f"({exc.__class__.__name__}: {exc}); "
                "falling back to the model-agnostic explainer."
            )

    def predict_fn(data: np.ndarray) -> np.ndarray:
        return np.ravel(fitted_pipeline.predict(data))

    masker = shap.maskers.Independent(X_background.to_numpy())
    return shap.Explainer(predict_fn, masker, feature_names=list(X_background.columns)), X_background


def compute_shap_values(fitted_pipeline, X: pd.DataFrame):
    """Compute SHAP values on X (also used as background for small data)."""
    explainer, X_used = _explainer_for(fitted_pipeline, X)
    return explainer(X_used if isinstance(X_used, pd.DataFrame) else X)


def generate_all_shap_plots(
    fitted_pipeline,
    X: pd.DataFrame,
    out_dir: Path,
    local_indices: list[int] | None = None,
) -> Path:
    """Produce every required SHAP figure into ``out_dir``.

    ``local_indices`` selects samples for the Local analysis (Waterfall);
    defaults to the first two rows.
    """
    import matplotlib.pyplot as plt
    import shap

    ensure_dir(out_dir)
    sv = compute_shap_values(fitted_pipeline, X)
    local_indices = local_indices or [0, 1]

    # Global: summary / beeswarm / bar importance
    for plot_name, plot_fn in {
        "shap_summary": lambda: shap.summary_plot(sv, X, show=False),
        "shap_beeswarm": lambda: shap.plots.beeswarm(sv, show=False),
        "shap_feature_importance": lambda: shap.plots.bar(sv, show=False),
    }.items():
        plt.figure()
        plot_fn()
        plt.tight_layout()
        plt.savefig(out_dir / f"{plot_name}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    # Global: dependence plot for every feature (shows non-linearity + interaction color)
    for feature in X.columns:
        plt.figure()
        shap.plots.scatter(sv[:, feature], color=sv, show=False)
        plt.tight_layout()
        safe = feature.replace("/", "_").replace(" ", "_")
        plt.savefig(out_dir / f"shap_dependence_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    # Local: waterfall for selected samples
    for idx in local_indices:
        plt.figure()
        shap.plots.waterfall(sv[idx], show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_waterfall_sample_{idx}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    # Feature interaction (tree models expose exact interaction values)
    try:
        inner = unwrap_pipeline(fitted_pipeline)
        model = inner.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        inter = explainer.shap_interaction_values(
            inner.named_steps["scaler"].transform(X)
        )
        mean_inter = np.abs(inter).mean(axis=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mean_inter, cmap="viridis")
        ax.set_xticks(range(len(X.columns)), X.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(X.columns)), X.columns)
        fig.colorbar(im, ax=ax, label="mean |SHAP interaction|")
        ax.set_title("SHAP Feature Interaction")
        fig.tight_layout()
        fig.savefig(out_dir / "shap_feature_interaction.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass  # non-tree models: interaction matrix not available

    return out_dir
