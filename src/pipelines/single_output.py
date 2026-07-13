"""Single-output regression pipeline (Pipelines 1 and 3).

For every (dataset, target, model, seed, fold) combination this module runs
cross-validation and returns/loggs a tidy results table. Used with:
  * Original datasets  -> Pipeline 1
  * Discrete datasets  -> Pipeline 3 (Top-3 models only)
"""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.data.loader import DatasetBundle
from src.models.registry import available_models, build_model
from src.pipelines.cross_validation import cross_validate_multi_seed
from src.tracking.mlflow_utils import log_cv_results


def run_single_output(
    bundle: DatasetBundle,
    models: list[str] | None = None,
    targets: list[str] | None = None,
    experiment_name: str | None = None,
    log_to_mlflow: bool = True,
) -> pd.DataFrame:
    """Run single-output CV for each target of a dataset.

    Parameters
    ----------
    bundle:
        Loaded dataset (original or discrete).
    models:
        Model names to run; defaults to every model in the config that is
        available in the environment.
    targets:
        Output columns to model; defaults to all Level-A outputs.
    """
    cfg = load_config()
    seeds = cfg["reproducibility"]["seeds"]
    requested = models if models is not None else cfg["models"]
    env_models = set(available_models())
    model_names = [m for m in requested if m in env_models]

    targets = targets if targets is not None else bundle.output_names

    all_rows: list[pd.DataFrame] = []
    for target in targets:
        y = bundle.Y[target]
        for model_name in model_names:
            # Base seed is used to build the estimator; CV re-seeds per run.
            estimator = build_model(model_name, seed=seeds[0])
            fold_df = cross_validate_multi_seed(estimator, bundle.X, y, seeds=seeds)
            fold_df.insert(0, "dataset", bundle.name)
            fold_df.insert(1, "mode", "single_output")
            fold_df.insert(2, "target", target)
            fold_df.insert(3, "model", model_name)
            all_rows.append(fold_df)

            if log_to_mlflow:
                log_cv_results(
                    experiment_name or f"{bundle.name}_single_output",
                    run_name=f"{model_name}__{target}",
                    fold_df=fold_df,
                    params={
                        "dataset": bundle.name,
                        "discrete": bundle.is_discrete,
                        "model": model_name,
                        "target": target,
                        "mode": "single_output",
                    },
                )

    return pd.concat(all_rows, ignore_index=True)
