"""Phase 6 - complementary importance methods.

  * Permutation importance      -> for the top models (model-agnostic)
  * Tree-based feature importance -> for tree models
  * TreeInterpreter             -> per-sample contribution decomposition
                                   (tree models only, optional dependency)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    fitted_pipeline, X: pd.DataFrame, y, seed: int = 42, n_repeats: int = 30
) -> pd.DataFrame:
    """Permutation importance on the full pipeline (scaling included)."""
    result = permutation_importance(
        fitted_pipeline,
        X.to_numpy(),
        np.asarray(y),
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def tree_feature_importance(fitted_pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Impurity-based feature importances from a fitted tree model."""
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise TypeError(f"{type(model).__name__} has no feature_importances_.")
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def treeinterpreter_contributions(fitted_pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Per-sample feature contributions via the treeinterpreter package.

    Works for RandomForest / ExtraTrees / DecisionTree regressors.
    """
    from treeinterpreter import treeinterpreter as ti

    scaler = fitted_pipeline.named_steps["scaler"]
    model = fitted_pipeline.named_steps["model"]
    X_scaled = scaler.transform(X.to_numpy())
    _, bias, contributions = ti.predict(model, X_scaled)
    contrib_df = pd.DataFrame(np.squeeze(contributions), columns=X.columns, index=X.index)
    contrib_df["bias"] = np.ravel(bias)
    return contrib_df
