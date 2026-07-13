"""Phase 6 - LIME local explanations for selected samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


def lime_explain_samples(
    fitted_pipeline,
    X: pd.DataFrame,
    out_dir: Path,
    sample_indices: list[int] | None = None,
    num_features: int | None = None,
) -> list:
    """Generate LIME explanations (figure + text) for selected samples."""
    from lime.lime_tabular import LimeTabularExplainer

    ensure_dir(out_dir)
    sample_indices = sample_indices or [0, 1]
    num_features = num_features or X.shape[1]

    explainer = LimeTabularExplainer(
        training_data=X.to_numpy(),
        feature_names=list(X.columns),
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )

    def predict_fn(data: np.ndarray) -> np.ndarray:
        return np.ravel(fitted_pipeline.predict(data))

    explanations = []
    for idx in sample_indices:
        exp = explainer.explain_instance(
            X.to_numpy()[idx], predict_fn, num_features=num_features
        )
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(out_dir / f"lime_sample_{idx}.png", dpi=150, bbox_inches="tight")
        with open(out_dir / f"lime_sample_{idx}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(f"{feat}: {weight:+.4f}" for feat, weight in exp.as_list()))
        explanations.append(exp)
    return explanations
