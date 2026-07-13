"""Multi-output regression pipeline (Pipelines 2 and 4).

All Level-A outputs of a dataset are predicted jointly. Estimators without
native multi-output support are wrapped in ``MultiOutputRegressor``
(one internal regressor per target, but trained/evaluated as one model).
"""

from __future__ import annotations

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data.loader import DatasetBundle
from src.models.registry import NATIVE_MULTIOUTPUT, available_models, build_model
from src.pipelines.cross_validation import cross_validate_multi_seed
from src.tracking.mlflow_utils import log_cv_results


def build_multioutput_model(name: str, seed: int) -> Pipeline:
    """Build a scaler+estimator pipeline that handles multi-output targets."""
    pipe = build_model(name, seed=seed)
    if name not in NATIVE_MULTIOUTPUT:
        # Wrap only the estimator step; the scaler stays shared.
        pipe.steps[-1] = ("model", MultiOutputRegressor(pipe.steps[-1][1]))
    return pipe


def run_multi_output(
    bundle: DatasetBundle,
    models: list[str] | None = None,
    targets: list[str] | None = None,
    experiment_name: str | None = None,
    log_to_mlflow: bool = True,
) -> pd.DataFrame:
    """Run multi-output CV for one dataset (metrics averaged over outputs).

    Note: sklearn's default multi-output R2/MAE are uniform averages across
    targets; per-target breakdowns for the Single-vs-Multi comparison
    (Phase 7) can be added by predicting per fold and splitting columns.
    """
    cfg = load_config()
    seeds = cfg["reproducibility"]["seeds"]
    requested = models if models is not None else cfg["models"]
    env_models = set(available_models())
    model_names = [m for m in requested if m in env_models]

    Y = bundle.Y[targets] if targets is not None else bundle.Y

    all_rows: list[pd.DataFrame] = []
    for model_name in model_names:
        estimator = build_multioutput_model(model_name, seed=seeds[0])
        fold_df = cross_validate_multi_seed(estimator, bundle.X, Y, seeds=seeds)
        fold_df.insert(0, "dataset", bundle.name)
        fold_df.insert(1, "mode", "multi_output")
        fold_df.insert(2, "target", "ALL")
        fold_df.insert(3, "model", model_name)
        all_rows.append(fold_df)

        if log_to_mlflow:
            log_cv_results(
                experiment_name or f"{bundle.name}_multi_output",
                run_name=f"{model_name}__multi",
                fold_df=fold_df,
                params={
                    "dataset": bundle.name,
                    "discrete": bundle.is_discrete,
                    "model": model_name,
                    "targets": ",".join(Y.columns),
                    "mode": "multi_output",
                },
            )

    return pd.concat(all_rows, ignore_index=True)
